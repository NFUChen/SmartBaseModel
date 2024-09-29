CODE_TEMPLATE = """
%s
%s
import json
print(f"<session>{json.dumps(globals(), default= str)}</session>")
"""
