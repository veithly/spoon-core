import os
import re
import csv
import pandas as pd
from datetime import timezone
from datetime import datetime
from dateutil.parser import parse
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import Field
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from loguru import logger
from spoon_ai.tools.base import BaseTool, ToolResult
from collections import defaultdict

class GitHubAnalysisTool(BaseTool):
    """GitHub Repository Analysis Tool"""
    name: str = Field(default="github_repo_analysis", description="Tool Name")
    description: str = Field(
        default="Retrieve and analyze GitHub repository Issues, PRs, Commits and collaboration data", 
        description="Tool Description"
    )
    parameters: dict = Field(
        default={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "branch": {"type": "string", "default": "master", "description": "Branch name"},
                "start_date": {"type": "string", "format": "date", "description": "Start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "format": "date", "description": "End date (YYYY-MM-DD)"},
                "force_fetch": {"type": "boolean", "default": False, "description": "Force refetch data"}
            },
            "required": ["owner", "repo", "start_date", "end_date"]
        },
        description="GitHub Analysis Parameters"
    )
    transport: Optional[Any] = Field(default=None, exclude=True)  # Exclude from schema
    client: Optional[Any] = Field(default=None, exclude=True)     # Exclude from schema
    data_dir: str = Field(default="./data/github", description="Data storage directory")
    current_start: datetime = Field(default=None, exclude=True)
    current_end: datetime = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        os.makedirs(self.data_dir, exist_ok=True)

    async def execute(self, **kwargs) -> ToolResult:
        """Execute analysis workflow"""
        try:
            logger.info(f"Starting analysis with parameters: {kwargs}")
            params = self._validate_params(kwargs)
            
            # Store current analysis time range
            self.current_start = params["raw_start"]
            self.current_end = params["raw_end"]
            
            # Configure GitHub client
            token = kwargs.get("access_token") or os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("GitHub access token not provided")
            self._configure_client(token)
            
            # Fetch data
            if kwargs.get("force_fetch") or not self._data_exists(params["repo"]):
                logger.info("Starting GitHub data retrieval...")
                await self._fetch_all_data(params)
            
            # Analyze data (using original date format)
            analysis = self._analyze_data({
                "repo": params["repo"],
                "start": params["raw_start"],
                "end": params["raw_end"]
            })
            
            if not analysis:
                logger.warning("Analysis result is empty")
                return ToolResult(error="Analysis result is empty")
                
            logger.success("Analysis completed")
            return ToolResult(output=analysis)
            
        except Exception as e:
            logger.exception("Error during analysis")
            return ToolResult(error=str(e))

    def _validate_params(self, params: Dict) -> Dict:
        """Validate and format input parameters"""
        try:
            # Convert user input as local time, transform to UTC timezone
            start_date = datetime.strptime(params["start_date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            end_date = datetime.strptime(params["end_date"], "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            
            return {
                "owner": params["owner"],
                "repo": params["repo"],
                "branch": params.get("branch", "master"),
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "raw_start": start_date,
                "raw_end": end_date
            }
        except ValueError as e:
            logger.error(f"Date parsing failed: {e}")
            raise ValueError("Date format should be YYYY-MM-DD") from e

    def _configure_client(self, token: str = None):
        """Configure GitHub client"""
        access_token = token or os.getenv("GITHUB_TOKEN")
        if not access_token:
            raise ValueError("GitHub access token required")

        self.transport = AIOHTTPTransport(
            url="https://api.github.com/graphql",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        self.client = Client(
            transport=self.transport,
            fetch_schema_from_transport=True
        )

    async def _fetch_all_data(self, params: dict):
        """Fetch all data"""
        try:
            await self._fetch_issues(params)
        except Exception as e:
            logger.error(f"Failed to fetch Issues: {str(e)}")
        
        try:
            await self._fetch_pull_requests(params)
        except Exception as e:
            logger.error(f"Failed to fetch Pull Requests: {str(e)}")
        
        try:
            await self._fetch_commits(params)
        except Exception as e:
            logger.error(f"Failed to fetch Commits: {str(e)}")

    async def _fetch_issues(self, params: dict):
        """Fetch Issues data (client-side filtering)"""
        query = gql("""
        query ($owner: String!, $repo: String!, $cursor: String) {
          repository(owner: $owner, name: $repo) {
            issues(
              first: 100
              after: $cursor
              filterBy: {
                states: [OPEN, CLOSED]
              }
            ) {
              edges {
                cursor
                node {
                  createdAt
                  state
                  url
                  author { login }
                  comments(first: 100) {
                    nodes {
                      createdAt
                      author { login }
                      url
                    }
                  }
                }
              }
            }
          }
        }
        """)
        
        variables = {
            "owner": params["owner"],
            "repo": params["repo"],
            "cursor": None
        }

        async for batch in self._paginate_query(query, variables, "repository.issues.edges"):
            for edge in batch:
                # Filter by time range
                created_at = pd.to_datetime(edge["node"]["createdAt"]).tz_convert('UTC')
                if created_at >= self.current_start and created_at <= self.current_end:
                    self._process_issue(edge["node"], params["repo"])

    def _process_issue(self, node: dict, repo: str):
        """Process single Issue (with time validation)"""
        created_at = pd.to_datetime(node["createdAt"]).tz_convert('UTC')
        
        # Check time range
        if created_at < self.current_start or created_at > self.current_end:
            logger.debug(f"Skipping Issue outside time range: {node['url']}")
            return
        
        issue_data = {
            "username": node["author"]["login"] if node["author"] else "",
            "state": node["state"],
            "created_at": created_at.isoformat(),
            "url": node["url"]
        }
        self._save_csv(f"{repo}_issue.csv", issue_data)

    async def _fetch_pull_requests(self, params: dict):
        """Fetch Pull Requests data (client-side filtering)"""
        query = gql("""
        query ($owner: String!, $repo: String!, $cursor: String) {
          repository(owner: $owner, name: $repo) {
            pullRequests(
              first: 100
              after: $cursor
              states: [OPEN, CLOSED, MERGED]
              orderBy: {field: CREATED_AT, direction: DESC}
            ) {
              edges {
                cursor
                node {
                  createdAt
                  state
                  url
                  author { login }
                  comments(first: 100) {
                    nodes {
                      createdAt
                      author { login }
                      url
                    }
                  }
                }
              }
            }
          }
        }
        """)
        
        variables = {
            "owner": params["owner"],
            "repo": params["repo"],
            "cursor": None
        }

        async for batch in self._paginate_query(query, variables, "repository.pullRequests.edges"):
            for edge in batch:
                # Filter by time range
                created_at = pd.to_datetime(edge["node"]["createdAt"]).tz_convert('UTC')
                if created_at >= self.current_start and created_at <= self.current_end:
                    self._process_pr(edge["node"], params["repo"])

    def _process_pr(self, node: dict, repo: str):
        """Process single PR"""
        pr_data = {
            "username": node["author"]["login"] if node["author"] else "",
            "state": node["state"],
            "created_at": node["createdAt"],
            "url": node["url"]
        }
        self._save_csv(f"{repo}_pullrequest.csv", pr_data)

        # Process comments
        for comment in node["comments"]["nodes"]:
            if comment["author"]:
                comment_data = {
                    "username": comment["author"]["login"],
                    "source": "PR",
                    "comment_time": comment["createdAt"],
                    "url": comment["url"]
                }
                self._save_csv(f"{repo}_comment.csv", comment_data)

    async def _fetch_commits(self, params: dict):
        """Fetch Commit data (client-side filtering)"""
        query = gql("""
        query ($owner: String!, $repo: String!, $branch: String!, $cursor: String) {
          repository(owner: $owner, name: $repo) {
            ref(qualifiedName: $branch) {
              target {
                ... on Commit {
                  history(
                    first: 100
                    after: $cursor
                  ) {
                    edges {
                      node {
                        message
                        url
                        author {
                          name
                          date
                        }
                        committedDate
                      }
                      cursor
                    }
                  }
                }
              }
            }
          }
        }
        """)
        
        variables = {
            "owner": params["owner"],
            "repo": params["repo"],
            "branch": params["branch"],
            "cursor": None
        }

        async for batch in self._paginate_query(query, variables, "repository.ref.target.history.edges"):
            for edge in batch:
                # Filter by time range
                commit_date = pd.to_datetime(edge["node"]["committedDate"]).tz_convert('UTC')
                if commit_date >= self.current_start and commit_date <= self.current_end:
                    self._process_commit(edge["node"], params["repo"])

    def _process_commit(self, node: dict, repo: str):
        """Process single Commit"""
        commit_data = {
            "username": node["author"]["name"],
            "commit_time": node["author"]["date"],
            "message": node["message"],
            "url": node["url"]
        }
        self._save_csv(f"{repo}_master_commit.csv", commit_data)

    async def _paginate_query(self, query, variables, data_path) -> AsyncGenerator:
        """Paginate query (with safety limits)"""
        max_pages = 20  # Prevent infinite loops
        page_count = 0
        
        while page_count < max_pages:
            page_count += 1
            
            try:
                # Copy variables to prevent modifying original
                query_vars = variables.copy()
                
                # Execute query
                result = await self.client.execute_async(query, variable_values=query_vars)
                edges = self._nested_get(result, data_path)
                
                if not edges:
                    logger.debug("No more data")
                    return
                    
                logger.debug(f"Retrieved page {page_count}, {len(edges)} records")
                yield edges
                
                # Update cursor
                if edges and edges[-1].get("cursor"):
                    variables["cursor"] = edges[-1].get("cursor")
                else:
                    return  # No more data
                    
            except Exception as e:
                logger.error(f"Error during query: {str(e)}")
                return
            
        else:
            logger.warning(f"Reached maximum pagination limit of {max_pages} pages")

    def _nested_get(self, data: dict, path: str) -> list:
        """Nested data retrieval"""
        keys = path.split(".")
        for key in keys:
            if not data:
                return []
            data = data.get(key, {})
        return data if isinstance(data, list) else []

    def _save_csv(self, filename: str, data: dict):
        """Save data to CSV"""
        path = os.path.join(self.data_dir, filename)
        fields = list(data.keys())
        XCsv.append_dict(path, data, fields)

    def _analyze_data(self, params: dict) -> dict:
        """Perform data analysis"""
        analysis = {}
        
        # Issues analysis
        issue_df = self._load_csv(f"{params['repo']}_issue.csv")
        if not issue_df.empty:
            filtered = self._filter_dates(issue_df, params["start"], params["end"])
            analysis["issues"] = {
                "opened": self._count_by(filtered, "state", "OPEN"),
                "closed": self._count_by(filtered, "state", "CLOSED")
            }

        # PRs analysis
        pr_df = self._load_csv(f"{params['repo']}_pullrequest.csv")
        if not pr_df.empty:
            filtered = self._filter_dates(pr_df, params["start"], params["end"])
            analysis["pull_requests"] = {
                "opened": self._count_by(filtered, "state", "OPEN"),
                "closed": self._count_by(filtered, "state", "CLOSED"),
                "merged": self._count_by(filtered, "state", "MERGED")
            }

        # Comments analysis
        comment_df = self._load_csv(f"{params['repo']}_comment.csv")
        if not comment_df.empty:
            filtered = self._filter_dates(comment_df, params["start"], params["end"])
            analysis["comments"] = filtered.groupby("username").size().to_dict()

        # Collaborators analysis
        coauthor_data = self._analyze_coauthors(params["repo"], params["start"], params["end"])
        analysis["coauthors"] = coauthor_data

        return analysis

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV data"""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
        elif "comment_time" in df.columns:
            df["comment_time"] = pd.to_datetime(df["comment_time"])
        elif "commit_time" in df.columns:
            df["commit_time"] = pd.to_datetime(df["commit_time"])
        return df

    def _filter_dates(self, df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        """Filter data by date (precise timezone handling)"""
        # Determine date column
        date_col = None
        for col in ["created_at", "comment_time", "commit_time"]:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            logger.warning("Cannot find date column for filtering")
            return df
        
        try:
            # Ensure data timezone is unified to UTC
            # Check if date already has timezone info
            datetime_series = pd.to_datetime(df[date_col])
            
            # Add UTC timezone if not present
            if datetime_series.dt.tz is None:
                df[date_col] = datetime_series.dt.tz_localize('UTC')
            else:
                # Already has timezone, convert to UTC
                df[date_col] = datetime_series
                
            logger.debug(f"Data time range: {df[date_col].min()} - {df[date_col].max()}")
            
            # Precise filtering
            mask = (df[date_col] >= start) & (df[date_col] <= end)
            filtered = df.loc[mask]
            logger.info(f"Filtered {len(filtered)}/{len(df)} valid records")
            return filtered
            
        except Exception as e:
            logger.error(f"Date filtering failed: {str(e)}")
            raise ValueError(f"Date processing error: {str(e)}") from e

    def _count_by(self, df: pd.DataFrame, column: str, value: str) -> dict:
        """Count by column value"""
        if df.empty or column not in df.columns:
            return {}
        return df[df[column] == value].groupby("username").size().to_dict()

    def _analyze_coauthors(self, repo: str, start: datetime, end: datetime) -> dict:
        """Analyze co-authors"""
        path = os.path.join(self.data_dir, f"{repo}_master_commit.csv")
        if not os.path.exists(path):
            return {}

        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        
        data = self._parse_commits(content)
        return self._count_coauthors(data, start, end)

    def _parse_commits(self, content: str) -> list:
        """Parse commit information"""
        pattern = r'(.*?),(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}),.*?Co-authored-by: (.*?) <.*?>'
        matches = re.findall(pattern, content, re.DOTALL)
        return [
            (m[0], parse(m[1]), m[2])
            for m in matches
        ]

    def _count_coauthors(self, data: list, start: datetime, end: datetime) -> dict:
        """Count co-author frequency"""
        counts = defaultdict(lambda: defaultdict(int))
        
        for author, date, coauthor in data:
            if start <= date <= end:
                counts[author][coauthor] += 1
                
        return {k: dict(v) for k, v in counts.items()}

    def _data_exists(self, repo: str) -> bool:
        """Check if data exists"""
        required_files = [
            f"{repo}_issue.csv",
            f"{repo}_pullrequest.csv",
            f"{repo}_comment.csv",
            f"{repo}_master_commit.csv"
        ]
        return all(os.path.exists(os.path.join(self.data_dir, f)) for f in required_files)


class XCsv:
    """CSV data operation utility class"""
    @staticmethod
    def append_dict(filename: str, data: dict, fields: list):
        """Append data to CSV file"""
        file_exists = os.path.exists(filename)
        mode = "a" if file_exists else "w"
        
        with open(filename, mode, newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    @staticmethod
    def create_csv(filename: str, fields: list):
        """Create new CSV file"""
        with open(filename, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()