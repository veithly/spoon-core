"""Integration tests for the NeoFS client."""

import os
import time
import uuid
import json

import pytest
from dotenv import load_dotenv

from spoon_ai.neofs.client import NeoFSClient
from spoon_ai.neofs.models import (
    Attribute,
    Bearer,
    ContainerPostInfo,
    Record,
    Rule,
    SearchFilter,
    SearchRequest,
    Target,
)

load_dotenv()

pytestmark = pytest.mark.skipif(
    not all(
        os.getenv(key)
        for key in [
            "NEOFS_BASE_URL",
            "NEOFS_OWNER_ADDRESS",
            "NEOFS_PRIVATE_KEY_WIF",
        ]
    ),
    reason="Missing required environment variables for NeoFS integration tests",
)


@pytest.fixture(scope="module")
def neofs_client() -> NeoFSClient:
    """Provide a configured NeoFS client instance."""

    return NeoFSClient()


@pytest.fixture(scope="module")
def container_token(neofs_client: NeoFSClient) -> str:
    """Generate a bearer token for container operations."""

    rule = Rule(verb="PUT", containerId="")
    bearer = Bearer(name="container-token", container=rule)
    tokens = neofs_client.create_bearer_tokens([bearer])
    return tokens[0].token


def test_get_network_info(neofs_client: NeoFSClient) -> None:
    info = neofs_client.get_network_info()
    assert info.epoch_duration > 0
    assert info.storage_price > 0


def test_get_balance(neofs_client: NeoFSClient) -> None:
    balance = neofs_client.get_balance()
    assert balance.address == neofs_client.owner_address
    assert int(balance.value) >= 0


@pytest.mark.skip(reason="TODO: Clarify REST gateway signing requirements for /v1/containers")
def test_container_lifecycle(neofs_client: NeoFSClient, container_token: str) -> None:
    container_name = f"test-container-{uuid.uuid4()}"
    container_info = ContainerPostInfo(
        containerName=container_name,
        placementPolicy="REP 3",
        basicAcl="public-read-write",
        attributes=[Attribute(key="test-key", value="test-value")],
    )

    created_container = neofs_client.create_container(container_info, container_token)
    assert created_container.container_name == container_name
    assert created_container.owner_id == neofs_client.owner_address
    container_id = created_container.container_id

    time.sleep(5)

    try:
        fetched_container = neofs_client.get_container(container_id)
        assert fetched_container.container_id == container_id
        assert fetched_container.container_name == container_name

        my_containers = neofs_client.list_containers(neofs_client.owner_address)
        assert any(container.container_id == container_id for container in my_containers.containers)

    finally:
        delete_response = neofs_client.delete_container(container_id)
        assert delete_response.success is True


@pytest.fixture(scope="function")
def temporary_container(neofs_client: NeoFSClient, container_token: str) -> str:
    pytest.skip("TODO: Clarify REST gateway signing requirements for /v1/containers and re-enable e2e tests")

    container_name = f"temp-test-container-{uuid.uuid4()}"
    container_info = ContainerPostInfo(
        containerName=container_name,
        placementPolicy="REP 3",
        basicAcl="public-read-write",
    )
    created_container = neofs_client.create_container(container_info, container_token)

    time.sleep(5)

    yield created_container.container_id

    try:
        neofs_client.delete_container(created_container.container_id)
    except Exception as error:  # pragma: no cover - best-effort cleanup
        print(f"Failed to cleanup container {created_container.container_id}: {error}")


@pytest.mark.skip(reason="TODO: Clarify REST gateway signing requirements for /v1/containers")
def test_object_lifecycle(neofs_client: NeoFSClient, temporary_container: str) -> None:
    container_id = temporary_container

    record = Record(
        action="ALLOW",
        operation="PUT",
        filters=[],
        targets=[Target(role="USER", keys=[neofs_client.owner_address])],
    )
    bearer = Bearer(name="object-put-token", object=[record])
    tokens = neofs_client.create_bearer_tokens([bearer])
    upload_token = tokens[0].token

    file_content = b"Hello, NeoFS!"
    file_name = f"test-file-{uuid.uuid4()}.txt"
    attributes = {"FileName": file_name, "Description": "A test file"}

    upload_address = neofs_client.upload_object(
        container_id,
        upload_token,
        file_content,
        attributes=attributes,
        expiration_duration="24h",
    )
    object_id = upload_address.object_id
    assert object_id is not None

    time.sleep(5)

    get_record = Record(action="ALLOW", operation="GET", filters=[], targets=[])
    search_record = Record(action="ALLOW", operation="SEARCH", filters=[], targets=[])
    bearer_get = Bearer(name="object-get-token", object=[get_record])
    bearer_search = Bearer(name="object-search-token", object=[search_record])

    tokens_get_search = neofs_client.create_bearer_tokens([bearer_get, bearer_search])
    download_token = next(token.token for token in tokens_get_search if token.name == "object-get-token")
    search_token = next(token.token for token in tokens_get_search if token.name == "object-search-token")

    try:
        header_response_id = neofs_client.get_object_header_by_id(container_id, object_id, download_token)
        assert header_response_id.status_code == 200
        assert int(header_response_id.headers["Content-Length"]) == len(file_content)
        retrieved_attrs = json.loads(header_response_id.headers["X-Attributes"])
        assert retrieved_attrs["FileName"] == file_name

        download_response_id = neofs_client.download_object_by_id(
            container_id,
            object_id,
            download_token,
            download=True,
        )
        assert download_response_id.status_code == 200
        assert download_response_id.content == file_content

        search_filter = SearchFilter(key="FileName", value=file_name, match="MatchStringEqual")
        search_request = SearchRequest(filters=[search_filter], attributes=["FileName"])
        search_result = neofs_client.search_objects(container_id, search_token, search_request)

        assert len(search_result.objects) > 0
        found_object = search_result.objects[0]
        assert found_object.object_id == object_id
        assert found_object.attributes["FileName"] == file_name

        header_response_attr = neofs_client.get_object_header_by_attribute(
            container_id,
            "FileName",
            file_name,
            download_token,
        )
        assert header_response_attr.status_code == 200

        download_response_attr = neofs_client.download_object_by_attribute(
            container_id,
            "FileName",
            file_name,
            download_token,
            download=True,
        )
        assert download_response_attr.status_code == 200
        assert download_response_attr.content == file_content

    finally:
        delete_response = neofs_client.delete_object(container_id, object_id)
        assert delete_response.success is True
