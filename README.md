# Violence_Recognition
Online and real-time violence recognition


## Data preparation
Preprocessing code from an official RWF-2000 github (https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)


## How to train
1. Choose model first. ( model / model_no_att )   
 -- Comment or Uncomment line 13 - 14 from main.py  
 -- Model without attention is good trade-off model in practice.   
 

2. 
```
sh train.sh
```



## Demo Video (trained on RWF-2000)
<img src="figures/three.gif" width="640" height="360"/>

## Demo on UCF Crimes (trained on RWF-2000)
<img src="figures/final.gif" width="640" height="360"/>

```
"A" in blue : model's detection result.
"B" in red : labels provided.
"C" in green : labels we made.
```

