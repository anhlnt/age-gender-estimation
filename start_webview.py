import webview
from demo import getScreenSize

SCREENSIZE = getScreenSize()
webview.create_window('VTM AI', url='http://localhost', x=SCREENSIZE[0] // 2, y=0, width=SCREENSIZE[0] // 2, height=SCREENSIZE[1])
webview.start()