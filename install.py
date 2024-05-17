import launch

if not launch.is_installed('py-cpuinfo'):
    launch.run_pip('install py-cpuinfo')
