module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "source env/bin/activate && uv pip install gradio colorama",
        path: "app"
      }
    },
    {
      method: "input",
      params: {
        title: "Update completed",
        description: "Click Start to launch."
      }
    }
  ]
}
