from IPython.display import display, HTML


def print_tile(key, value, size="h1"):
  cleaned_value = str(value).replace("<", "").replace(">", "")
  return display(HTML(f"""<p style="color:grey">{key}</p><{size} font-size: 3em>{cleaned_value}</{size}>"""))