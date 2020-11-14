@echo off
call D:\App\Anaconda3\Scripts\activate.bat D:\App\Anaconda3
call conda activate irs
set path=D:\App\Anaconda3\envs\irs;%path%
D:
cd "D:\ISS\MTech2020\ISY5003\Practice Module\3DMapping\visualization\bin"

start start_backend_services.bat

start start_3d_visualizer.bat

call start_map_viewer.bat

