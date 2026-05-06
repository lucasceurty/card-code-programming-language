_str_a = (chr((10 * 7)) + (chr(((10 * 10) + 5)) + (chr((((10 + 10) * 6) + 2)) + chr((((10 + 10) * 6) + 2)))))
_str_b = (chr(((10 * 6) + 6)) + (chr((((10 * 10) + 10) + 7)) + (chr((((10 + 10) * 6) + 2)) + chr((((10 + 10) * 6) + 2)))))
for _ctr in range(1, ((10 + 5)) + 1):
    if (((_ctr % 3) == 0) and ((_ctr % 5) == 0)):
        print((_str_a + _str_b))
    else:
        if ((_ctr % 3) == 0):
            print(_str_a)
        else:
            if ((_ctr % 5) == 0):
                print(_str_b)
            else:
                print(str(_ctr))