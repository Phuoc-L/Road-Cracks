# Road Damage Detection

## Models

| Model | Description |
|-------|-------------|
| Base_SGD.pt | Base model with SGD optimizer |
| Base_AdamW.pt | Base model with AdamW optimizer |
| Base_AdamWV2.pt | Base model with optimized AdamW parameters |
| Custom_Model.pt | First custom architecture model |
| Custom_Model2.pt | Second custom architecture model |

## Datasets

- **RDD2022.zip** - Road Damage Detection Dataset
- **COCO2017.zip** - Common Objects in Context Dataset

## Code Files

- `road_obstacles.ipynb`: Combine and split datasets, train and evaluate models
- `Object_Detection_Eval_Colab.ipynb`: Compare different object detection models
- `requirements.txt`: Project dependencies
- `export.py`: Export models to different formats
- `yolov8_custom.yaml`: First custom architecture configuration
- `yolov8_custom2.yaml`: Second custom architecture configuration

## Running the Frontend Application

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Launching the Application

```bash
# Start the application
python app.py
```
