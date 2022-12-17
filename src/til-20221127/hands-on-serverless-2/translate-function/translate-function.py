"""AWS Lambdaで翻訳実行するためのモジュール."""
import json
import logging

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """AWS Lambdaで翻訳実行するエントリポイント."""
    _logger.info(event)

    return {"statusCode": 200, "body": json.dumps("Hello Hands on world!")}
