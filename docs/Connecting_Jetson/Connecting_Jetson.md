# Connecting to the Jetson via Terminal

There are two ways to connect to the Jetson via SSH. 

* Serial: uses a micro-usb cable to connect
    * Pros: Easier to use; less finicky
    * Cons: Multiple people can't connect at once; requires a micro-usb cable
* Internet: uses the wifi to connect
    * Pros: Multiple people can connect to one Jetson; no cable required
    * Cons: Can be slow when too many people connect at once; requires wifi dongle

In general, you should use the serial method unless you need to connect multiple people to one Jetson or can't get a micro-usb cable.

## Serial
### Windows

