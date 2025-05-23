# sudo apt update
# sudo apt install -y python3-gi gir1.2-gst-rtsp-server-1.0 \
#                     gstreamer1.0-tools \
#                     gstreamer1.0-plugins-base \
#                     gstreamer1.0-plugins-good \
#                     gstreamer1.0-plugins-bad \
#                     gstreamer1.0-plugins-ugly \
#                     gstreamer1.0-openni2

from gi.repository import GObject, Gst, GstRtspServer

Gst.init(None)

class OpenNI2Factory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super().__init__()

        self.launch_string = (
            'openni2src ! '
            'video/x-raw,format=RGB,framerate=30/1,width=640,height=480 ! '
            'videoconvert ! '
            'x264enc tune=zerolatency bitrate=800 speed-preset=superfast ! '
            'rtph264pay name=pay0 pt=96 config-interval=1'
        )

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.set_shared(True)

if __name__ == '__main__':
    
    server = GstRtspServer.RTSPServer()
    server.props.service = '8554'
    mounts = server.get_mount_points()

    factory = OpenNI2Factory()
    mounts.add_factory('/stream', factory)  # URL: rtsp://192.168.0.217:8554/stream

    server.attach(None)
    print(">>> RTSP OpenNI2 corriendo en rtsp://0.0.0.0:8554/stream")
    loop = GObject.MainLoop()
    loop.run()
