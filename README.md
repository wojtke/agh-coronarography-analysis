# corflow

Dataset Dir Structure

```
.
├── frame_selection/
│   ├── dataset.csv
│   ├── dicoms/
│   │   └── ...
│   └── metadata/
│       └── ...
├── series_selection/
│   ├── rejected/
│   │   └── ...
│   ├── accepted/
│   │   └── ...
│   ├── accepted.csv
│   └── rejected.csv
├── framerejectionfeedback.csv
├── imagerejectionfeedback.csv
└── imagefeedback.csv
```



TODO
- load new frame selection and series selection, prepare sh script
- load new rca/lca from karol
- eval dominance by 