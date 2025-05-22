from bitalino import BITalino

mac_address = "30:16:12:21:98:74"  # Sustituye por la MAC real
device = BITalino(mac_address)

device.start(samplingRate=10, channels=[0])
data = device.read(100)
device.stop()
device.close()
