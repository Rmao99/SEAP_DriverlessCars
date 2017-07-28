from gopigo import *

enable_encoders()
enc_tgt(1,1,18)
while read_enc_status():
	print "in reading encorder status"
	set_speed(45)	
	fwd()
disable_encoders()
stop()
