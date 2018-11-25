for experiment in `ls experiments`
do
    cd experiments/$experiment
    python main.py -input=../../inputs/ramona-color.png -source=../../inputs/starry-night.png
    cd ../../
done

