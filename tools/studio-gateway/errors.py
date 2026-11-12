"""Public-safe gateway failure values; never include credentials or paths."""

class GatewayError(Exception):
    def __init__(self, status: int, message: str, code: str = "gateway_error"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code

