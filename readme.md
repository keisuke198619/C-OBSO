## Compute GVRNN and OBSO
 
## Requirements

* python 3.6 
* To install requirements:

```setup
pip install -r requirements.txt
```
## Data
* J-league dataset was purchased and should not be shared.
* `data_jleague` folder was set at the same level as this folder.
* Instead, you can set e.g., Metrica data. See also the tutorial code: `https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking`.

## Trajectory prediction
* Run `run.sh` for training and test.
* `main_GVRNN.py`, `features.py`, `hidden_role_learning.py`, `preprocessing.py`, `sequencing.py`, `utilities.py` are related files.
* learnt weights and predicted trajectory are at `VRNN_Jleague_data`.
* `VRNN_Jleague_data` folder should be set at the same level as this folder.
* In particular, predicted trajectory is at `VRNN_Jleague_data\WEIGHTS\...`

## Compute OBSO
* Run `calculate_obso.py` for computing OBSO.
* `Metrica_IO`, `Metrica_Viz`, `Metrica_Velocities`, `Metrica_PitchControl`, `Metrica_EPV`, `obso_player.py`, `third_party.py`, `Transition_gauss.csv`, `EPV_grid.csv` are related files.
