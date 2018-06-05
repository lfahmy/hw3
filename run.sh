python train.py --optimizer ADAM --cuda
python train.py --optimizer SGD --cuda
python train.py --optimizer ADAM --cuda --isDropOut
python train.py --optimizer SGD --cuda --isDropOut
python test.py --optimizer ADAM --cuda
python test.py --optimizer SGD --cuda
python test.py --optimizer ADAM --cuda --isDropOut
python test.py --optimizer SGD --cuda --isDropOut
