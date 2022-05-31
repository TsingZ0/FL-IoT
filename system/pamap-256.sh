nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedAvg-FT -did 4 > pamap-256-FedAvg-FT.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedProx-FT -did 4 -mu 1e-5 > pamap-256-FedProx-FT.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo Local -did 4 > pamap-256-Local.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedHome -did 1 > pamap-256-FedHome.out 2>&1 &

nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPruneW -did 1 -tk 10 > pamap-256-FedPruneW-tk=10.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPruneW -did 1 -tk 20 > pamap-256-FedPruneW-tk=20.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPruneW -did 2 -tk 30 > pamap-256-FedPruneW-tk=30.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPruneW -did 2 -tk 40 > pamap-256-FedPruneW-tk=40.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPruneW -did 3 -tk 50 > pamap-256-FedPruneW-tk=50.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPruneW -did 3 -tk 60 > pamap-256-FedPruneW-tk=60.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPruneW -did 4 -tk 70 > pamap-256-FedPruneW-tk=70.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPruneW -did 4 -tk 80 > pamap-256-FedPruneW-tk=80.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPruneW -did 4 -tk 90 > pamap-256-FedPruneW-tk=90.out 2>&1 &

nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPrune -did 1 -tk 10 -gr 20 > pamap-256-FedPrune-tk=10.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPrune -did 2 -tk 20 -gr 20 > pamap-256-FedPrune-tk=20.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPrune -did 3 -tk 30 -gr 20 > pamap-256-FedPrune-tk=30.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPrune -did 6 -tk 40 -gr 20 > pamap-256-FedPrune-tk=40.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPrune -did 6 -tk 60 -gr 20 > pamap-256-FedPrune-tk=60.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPrune -did 6 -tk 70 -gr 20 > pamap-256-FedPrune-tk=70.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPrune -did 7 -tk 80 -gr 20 > pamap-256-FedPrune-tk=80.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 9 -jr 1 -nb 12 -data pamap-256 -m harcnn -algo FedPrune -did 7 -tk 90 -gr 20 > pamap-256-FedPrune-tk=90.out 2>&1 &