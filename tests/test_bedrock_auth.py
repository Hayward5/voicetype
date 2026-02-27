"""
Bedrock authentication tests - API-key vs IAM branches.
Mocks requests and boto3 to verify correct behavior.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock boto3 module before tests try to patch it
if 'boto3' not in sys.modules:
    sys.modules['boto3'] = MagicMock()


class TestBedrockAuth(unittest.TestCase):
    """Tests for Bedrock API-key and IAM authentication modes."""

    def _create_mock_settings(self, bedrock_api_key: str | None = None):
        """Create a mock settings object."""
        mock_settings = MagicMock()
        mock_settings.get_api_key = MagicMock(
            side_effect=lambda key: {
                "bedrock": bedrock_api_key,
            }.get(key)
        )
        mock_settings.get_config = MagicMock(
            return_value={
                "llmProvider": "bedrock",
                "bedrockRegion": "us-east-1",
                "llmModel": "amazon.nova-lite-v1:0",
                "contextAware": False,
                "systemPrompt": "sys",
            }
        )
        return mock_settings

    @patch("requests.get")
    @patch("requests.post")
    def test_api_key_list_models_uses_bearer_and_params(self, mock_post, mock_get):
        """API-key mode: list_models uses Authorization Bearer and correct URL/params."""
        from core.llm import LLMProcessor

        api_key = "test-api-key-12345"
        mock_settings = self._create_mock_settings(bedrock_api_key=api_key)

        # Mock requests.get response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "modelSummaries": [
                {"modelId": "model-a"},
                {"modelId": "model-b"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        processor = LLMProcessor(mock_settings)
        result = processor.list_models("bedrock")

        # Verify requests.get was called
        self.assertTrue(mock_get.called)

        # Verify URL contains foundation-models endpoint
        call_args = mock_get.call_args
        url = call_args[0][0]
        self.assertIn("bedrock.us-east-1.amazonaws.com/foundation-models", url)

        # Verify Authorization Bearer header
        headers = call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], f"Bearer {api_key}")

        # Verify params
        params = call_args[1]["params"]
        self.assertEqual(params["byOutputModality"], "TEXT")

    @patch("requests.get")
    @patch("requests.post")
    def test_api_key_polish_uses_bearer_content_type_and_json_payload(
        self, mock_post, mock_get
    ):
        """API-key mode: polish uses Authorization Bearer + Content-Type, correct URL, JSON payload keys."""
        from core.llm import LLMProcessor

        api_key = "test-api-key-12345"
        mock_settings = self._create_mock_settings(bedrock_api_key=api_key)

        # Mock requests.post response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"content": [{"text": "polished text"}]}}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        processor = LLMProcessor(mock_settings)
        result = processor.polish("raw text here")

        # Verify requests.post was called
        self.assertTrue(mock_post.called)

        # Verify URL contains bedrock-runtime endpoint
        call_args = mock_post.call_args
        url = call_args[0][0]
        self.assertIn("bedrock-runtime.us-east-1.amazonaws.com", url)
        self.assertIn("/model/amazon.nova-lite-v1:0/converse", url)

        # Verify Authorization Bearer + Content-Type headers
        headers = call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], f"Bearer {api_key}")
        self.assertEqual(headers["Content-Type"], "application/json")

        # Verify JSON payload keys
        body = call_args[1]["json"]
        self.assertIn("messages", body)
        self.assertIn("system", body)
        self.assertIn("inferenceConfig", body)

    @patch("boto3.client")
    @patch("requests.get")
    @patch("requests.post")
    def test_iam_mode_does_not_call_requests(self, mock_post, mock_get, mock_boto3):
        """IAM mode: does not call requests, uses boto3 client instead."""
        from core.llm import LLMProcessor

        # No API key = IAM mode
        mock_settings = self._create_mock_settings(bedrock_api_key=None)

        # Mock boto3 client
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {"modelId": "iam-model-a"},
                {"modelId": "iam-model-b"},
            ]
        }
        mock_boto3.return_value = mock_bedrock_client

        processor = LLMProcessor(mock_settings)
        result = processor.list_models("bedrock")

        # Verify requests.get was NOT called
        mock_get.assert_not_called()

        # Verify boto3.client was called
        mock_boto3.assert_called_once_with("bedrock", region_name="us-east-1")

        # Verify list_foundation_models was called
        mock_bedrock_client.list_foundation_models.assert_called_once_with(
            byOutputModality="TEXT"
        )

    @patch("boto3.client")
    @patch("requests.get")
    @patch("requests.post")
    def test_iam_mode_polish_does_not_call_requests(
        self, mock_post, mock_get, mock_boto3
    ):
        """IAM mode polish: does not call requests, uses boto3 client."""
        from core.llm import LLMProcessor

        # No API key = IAM mode
        mock_settings = self._create_mock_settings(bedrock_api_key=None)

        # Mock boto3 client
        mock_runtime_client = MagicMock()
        mock_runtime_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "iam polished text"}]}}
        }
        mock_boto3.return_value = mock_runtime_client

        processor = LLMProcessor(mock_settings)
        result = processor.polish("raw text here")

        # Verify requests.post was NOT called
        mock_post.assert_not_called()

        # Verify boto3.client was called with bedrock-runtime
        mock_boto3.assert_called_once_with("bedrock-runtime", region_name="us-east-1")

        # Verify converse was called
        mock_runtime_client.converse.assert_called_once()


if __name__ == "__main__":
    unittest.main()
