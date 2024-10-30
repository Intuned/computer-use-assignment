from IPython.display import display_html
from termcolor import colored


def prefixed_print(content, prefix=""):
    print(*[prefix + line for line in content.split("\n")], sep="\n")


def print_conversation(messages):
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        on_color = "on_green" if role == "user" else "on_red"
        color = "green" if role == "user" else "red"
        print(colored(f"[{role}]:", "black", on_color, ["bold"]))

        def print_content(content, prefix=""):
            if isinstance(content, str):
                print(colored(content, color))
                return

            for content_part in content:
                if isinstance(content_part, str):
                    print(colored(content_part, color))
                    continue
                c_type = content_part.get("type")
                if c_type == "text":
                    prefixed_print(colored(content_part.get("text"), color), prefix)
                elif c_type == "image":
                    prefixed_print(colored("Image", "yellow"), prefix)
                    html = f'<img src="data:image/png;base64,{content_part.get("source").get("data")}" width="500">'
                    display_html(html, raw=True)
                elif c_type == "base64":
                    prefixed_print(
                        colored(f"[{content_part.get("media_type")}]: <{content_part.get("data")[:10]}...>",
                                "yellow"), prefix)
                elif c_type == "tool_use":
                    prefixed_print(
                        colored(f"Using tool {content_part.get("name")} with input {content_part.get("input")}",
                                "cyan"), prefix)
                elif c_type == "tool_result":
                    prefixed_print(colored(f"Tool result:", "cyan"), prefix)
                    print_content(content_part.get("content"), prefix + "  ")

        print_content(content)
        print("\n")


def print_stuff(prefix, *args, **kwargs):
    # pass
    print(f"[{prefix}]", args, kwargs)


def noop(*_, **__):
    pass
