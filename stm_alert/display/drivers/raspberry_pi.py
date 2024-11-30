# /*****************************************************************************
# * | File        :	  lcdconfig.py
# * | Author      :   Waveshare team
# * | Function    :   Hardware underlying interface
# * | Info        :
# *----------------
# * | This version:   V1.0
# * | Date        :   2019-06-21
# * | Info        :   
# ******************************************************************************
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documnetation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to  whom the Software is
# furished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS OR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import os
import sys
import time
import spidev
import logging
import numpy as np
from gpiozero import *

class RaspberryPi:
    def __init__(self,bus=0,device=0,spi_freq=40000000,rst = 22,dc = 23,bl = 19,bl_freq=1000,i2c=None,i2c_freq=100000):
        self.INPUT = False
        self.OUTPUT = True
        
        self.SPEED  =spi_freq
        self.BL_freq=bl_freq
        
        #Initialize GPIO
        self.RST_PIN= self.gpio_mode(rst,self.OUTPUT)
        self.DC_PIN = self.gpio_mode(dc,self.OUTPUT)
        self.BL_PIN = self.gpio_pwm(bl)
        #self.no_cs = True

        #Initialize SPI
        self.SPI = spidev.SpiDev(bus, device)
        if self.SPI!=None :
            self.SPI.max_speed_hz = spi_freq
            self.SPI.mode = 0b00

    def gpio_mode(self,Pin,Mode,pull_up = None,active_state = True):
        if Mode:
            return DigitalOutputDevice(Pin,active_high = True,initial_value =False)
        else:
            return DigitalInputDevice(Pin,pull_up=pull_up,active_state=active_state)

    def digital_write(self, pin, value):
        if value:
            pin.on()
        else:
            pin.off()

    def digital_read(self, pin):
        return pin.value

    def delay_ms(self, delaytime):
        time.sleep(delaytime / 1000.0)

    def gpio_pwm(self,Pin):
        return PWMOutputDevice(Pin,frequency = self.BL_freq)
        
    def spi_writebyte(self, data):
        if self.SPI!=None :
            self.SPI.writebytes(data)
    def bl_DutyCycle(self, duty):
        # self._pwm.ChangeDutyCycle(duty)
        self.BL_PIN.value = duty / 100
        # os.system("gpio -g pwm 13 " + str(int(duty * 10.23)) )
        
    def bl_Frequency(self,freq):
        self.BL_PIN.frequency = freq
           
    def module_init(self):
        if self.SPI!=None :
            self.SPI.max_speed_hz = self.SPEED        
            self.SPI.mode = 0b00     
        return 0

    def module_exit(self):
        logging.debug("spi end")
        if self.SPI!=None :
            self.SPI.close()
        
        logging.debug("gpio cleanup...")
        self.digital_write(self.RST_PIN, 1)
        self.digital_write(self.DC_PIN, 0)   
        self.BL_PIN.close()
        time.sleep(0.001)