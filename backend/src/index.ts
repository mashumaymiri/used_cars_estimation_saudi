import express from "express";
import { PythonShell, PythonShellError } from "python-shell";
import cors from "cors";


const app = express();

app.use(cors())

app.get("/", cors(),(req: express.Request, res: express.Response): void => {
  const Make: string = req.query["Make"] as string;
  const Type: string = req.query["Type"] as string;
  const Year: number = parseInt(req.query["Year"] as unknown as string);
  const Origin: string = req.query["Origin"] as string;
  const Color: string = req.query["Color"] as string;
  const Options: string = req.query["Options"] as string;
  const Engine_Size: number = parseFloat(
    req.query["Engine_Size"] as unknown as string
  );
  const Fuel_Type: string = req.query["Fuel_Type"] as string;
  const Gear_Type: string = req.query["Gear_Type"] as string;
  const Mileage: number = parseInt(req.query["Mileage"] as unknown as string);
  const Region: string = req.query["Region"] as string//"Riyadh";

  const data: (string | number)[] = [
    Make,
    Type,
    Year,
    Origin,
    Color,
    Options,
    Engine_Size,
    Fuel_Type,
    Gear_Type,
    Mileage,
    Region,
  ];

  //   console.log(data);

  PythonShell.run(
    "run model.py",
    {
      mode: "text",
      pythonPath: "python",
      pythonOptions: ["-u"], // get print results in real-time
      scriptPath: __dirname + "/../" + "Model",
      args: data as string[],
    },
    function (
      err: PythonShellError | undefined,
      results: string[] | undefined
    ): void {
      if (err) {
        console.log(err);
        throw err;
      }
    //   console.log("results: %j", results);
      res.send(results);
    }
  );
});

app.listen(3000);
