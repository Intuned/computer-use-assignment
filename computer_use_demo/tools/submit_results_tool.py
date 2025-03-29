# type: ignore
from typing import ClassVar
from typing import Literal
from typing import Type
from typing import TypeVar

from anthropic.types.beta import BetaToolUnionParam
from pydantic import BaseModel

from .base import BaseAnthropicTool
from .base import ToolResult

T = TypeVar("T", bound=BaseModel)


class TerminateLoopToolResult(ToolResult):
    """
    A tool result that terminates the loop
    """

    def __init__(self, data: T):
        self.reason = "got results"
        self.data = data
        super().__init__()


class SubmitResultsTool(BaseAnthropicTool):
    """
    A tool that allows submitting results with a configurable input model and callback
    """

    name: ClassVar[Literal["submit_results"]] = "submit_results"

    def __init__(self, model_class: Type[T]):
        self.model_class = model_class
        # self.callback = callback
        super().__init__()

    async def __call__(self, **kwargs):
        # Parse input into the provided model class
        parsed_data = self.model_class.model_validate(kwargs)

        # Execute the callback with the parsed data
        # self.callback(parsed_data)

        return TerminateLoopToolResult(data=parsed_data)

    def to_params(self) -> BetaToolUnionParam:
        schema = {"name": self.name, "input_schema": self.model_class.model_json_schema()}
        return schema  # type: ignore
