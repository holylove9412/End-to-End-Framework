import subprocess
from pathlib import Path
from dotenv import load_dotenv
import swmmtoolbox.swmmtoolbox as swm
load_dotenv()
swmm_executable_path = 'D:/EPASWMM5.1.015/swmm5.exe'
nf = Path('D:/pythonProject4/FloodingRiskAssessment/network/smallexample')
subprocess.run(
    [
        swmm_executable_path,
        nf / "smallexample.inp",
        nf / "smallexample.rpt",
        nf / "smallexample.out",
    ]
)
head_out_timeseries=swm.extract(nf / "smallexample.out", ("node", "", "Hydraulic_head"))

