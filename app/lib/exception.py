from app.constants.status import Status


class AgentException(Exception):
    def __init__(self, code: Status, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(msg)
