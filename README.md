powershell admin
```
usbipd list
usbipd bind --busid x-x
usbipd attach --wsl --busid x-x
```

wsl
```
lsusb
ls /dev/input
```




```
python keyboard_test.py
python xbox_test.py --device /dev/input/event0
python test.py --controller keyboard
python test.py --controller xbox --device /dev/input/event0
```

Controller keybinds and sensitivity limits live in `controller/config.yaml`.


