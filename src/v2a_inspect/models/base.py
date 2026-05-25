from pydantic import BaseModel, ConfigDict


class SchemaModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def to_tool_string(self) -> str:
        return self.model_dump_json(
            indent=2,
            exclude_none=True,
            exclude_computed_fields=True,
        )

    def __str__(self) -> str:
        return self.to_tool_string()
