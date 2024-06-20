# SurfaceGPS: Multi-modal Localization for Facade Robots
```
ros2 launch surface_gps localizer.launch.py -s config_file:=.../src/surface_gps/config/config.yaml
```

```
ros2 service call /gps_avg surface_gps_interface/srv/AvgGPS "{filter_size: 10}"
```

- Move the `surface_gps_interface` package to `src` folder at the same level as `surface_gps`
- TODO: Ensure build process maintains current folder structure
