import express from "express";
import {PythonShell} from 'python-shell';

const app = express()

app.get("/", (req : express.Request, res: express.Response) => {
    console.log(req.query)
    PythonShell.run('../Model/run model.py', {
    mode: 'text',
    pythonPath: '../Model/run model.pyn',
    pythonOptions: ['-u'], // get print results in real-time
    scriptPath: '../Model/run model.py',
    args: ['value1', 'value2', 'value3']
  }, 
  function (err, results) {
        if (err) throw err;
        console.log('results: %j', results);
    });
    res.send("test")
})

app.listen(3000)