class ToolError(Exception):
    """Raised by a tool on bad input it can't recover from (unknown doc, bad regex...).

    The harness catches this and turns it into an error observation rather than
    letting it crash the agent loop.
    """
