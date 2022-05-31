nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedAvg-FT -did 1 > har-FedAvg-FT.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedProx-FT -did 1 -mu 1e-5 > har-FedProx-FT.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo Local -did 1 > har-Local.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedHome -did 1 > har-FedHome.out 2>&1 &

nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPruneW -did 1 -tk 10 > har-FedPruneW-tk=10.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPruneW -did 2 -tk 20 > har-FedPruneW-tk=20.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPruneW -did 2 -tk 30 > har-FedPruneW-tk=30.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPruneW -did 2 -tk 40 > har-FedPruneW-tk=40.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPruneW -did 2 -tk 50 > har-FedPruneW-tk=50.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPruneW -did 2 -tk 60 > har-FedPruneW-tk=60.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPruneW -did 3 -tk 70 > har-FedPruneW-tk=70.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPruneW -did 3 -tk 80 > har-FedPruneW-tk=80.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPruneW -did 3 -tk 90 > har-FedPruneW-tk=90.out 2>&1 &

nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPrune -did 1 -tk 10 -gr 20 > har-FedPrune-tk=10.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPrune -did 1 -tk 20 -gr 20 > har-FedPrune-tk=20.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPrune -did 2 -tk 30 -gr 20 > har-FedPrune-tk=30.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPrune -did 2 -tk 40 -gr 20 > har-FedPrune-tk=40.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPrune -did 3 -tk 60 -gr 20 > har-FedPrune-tk=60.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPrune -did 3 -tk 70 -gr 20 > har-FedPrune-tk=70.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPrune -did 4 -tk 80 -gr 20 > har-FedPrune-tk=80.out 2>&1 &
nohup python -u main.py -lr 0.01 -lbs 10 -nc 30 -jr 1 -nb 6 -data har -m harcnn -algo FedPrune -did 4 -tk 90 -gr 20 > har-FedPrune-tk=90.out 2>&1 &