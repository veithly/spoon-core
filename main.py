import argparse
import logging

from cli.commands import SpoonAICLI

logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="SpoonAI CLI")
    parser.add_argument('--server', action='store_true', help='Start the server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', default=8000, type=int, help='Server port')
    args = parser.parse_args()
    if args.server:
        raise NotImplementedError("Server mode is not implemented yet")

    else:
        cli = SpoonAICLI()
        cli.run()
if __name__ == "__main__":
    main()
        