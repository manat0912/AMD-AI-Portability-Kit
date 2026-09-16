module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        message: "source env/bin/activate && python app.py",
        path: "app",
        env: {
          HSA_ENABLE_DXG_DETECTION: "1"
        },
        on: [{
          event: "/http:\\/\\/[0-9.:]+/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    }
  ]
}
