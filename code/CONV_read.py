#!/anaconda3/python.app/Contents/MacOS/python
# -*- coding: utf-8 -*-
'''
put at the top of the file:
#!/anaconda3/python.app/Contents/MacOS/python
#!/Users/fmriadmin/anaconda2/bin/python

~~ Script for the CONV study ~~

TASK:

Participants, in pairs, are simultaneously scanned while they take turns chatting about 20 different topics.
There are 5 runs of 4 topic prompts, with later runs containing more intimate / personal prompts.

Timing of each run: (blank: 10 s) + 4 trials x [(intro: 10 s + conversation: 180 s + blank: 10 s) + (blank: 10 s) = 820 s = 13.67 mins

PRIOR TO RUNNING THE SCRIPT:

(1) System clocks on the control computers (or laptops running the script) need to be synced prior to running the experiment.
This can be done by either using the same NTP servers or by syncing to the same GPS time signal.

(2) (at Princeton) Make sure the matrix input one (1) is connected to the stimulus computer.
Otherwise, the stimulus computer will not receive any TTL pulse keypresses during the functional scans.

WHAT THE SCRIPT DOES:

(1) Transmit audio between distant sites using UDP. UDP hole punching is used to establish UDP connections with systems behind NAT.
UDP packets are uncompressed audio data appended with timestamps.

(2) Synchronizes task start on both control computers by having the two computers exchange timestamps and deriving a common start time from those timestamps.

(3) Uses PsychoPy for visual stimuli and keyboard events during the task.

NOTES:

- This script has only been tested on OS X 10.11 and OS X. 10.12.

- This script uses PsychoPy 3; have not tested using PsychoPy versions with python 2.7.

- The TTL pulse keypress at Princeton is the equal sign (=).

- Hardcoded variables can be found in the magicNumbers function. If you need to set parameters, that's where you should go.

- The last 2 digits of a participant ID denotes pair #.

- This script is capable of simple NAT traversal. If NATs are present and there are two firewalls, a static IP address is required at least on one end.

- With UDP, we may lose a few packages, but it shouldn't affect audio quality much.
To account for fluctuations in travel time / losses, the program uses a simple two-ended continuous audio buffer, with only occasional
health reports (nothing adaptive). You can control audio chunk and buffer size with command-line options --CHUNK and --BUFFER.

- Syncing start time across computers is done using a third UDP socket pair.

- We can check current NAT properties with stunclient using the input option --STUN.

CREDITS:

Audio link and task synchronization was made possible with the help of Adam Boncz.

Author: Lily Tsoi
Last modified: October 7, 2019

---

# TODO:
- length of the audio file should be length of run (or not? think it through)

'''

## -- import modules and provide python 2.7 / 3 compatibility --

# imports providing backward compatibility to 2.7
from __future__ import absolute_import, print_function, division, unicode_literals
import os, sys

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# for home testing only
sys.path.append('/anaconda3/lib/python3.6/site-packages')

# from builtins import range, map, bytes
from io import open
# other imports
import random, math, datetime
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
import pandas as pd
import argparse, socket, time, struct, subprocess, re, wave, csv, multiprocessing, pyaudio, string
from operator import sub

# Python 2.7 / 3 compatibility
# get python 3 flush arg behavior on 2.7 for print function
if sys.version_info[:2] < (3, 3):
    old_print = print

    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        file = kwargs.get('file', sys.stdout)
        if flush and file is not None:
            file.flush()
# get python 3 user input behavior
if hasattr(__builtins__, 'raw_input'):
    input = raw_input

# enter hardcoded values here
def magicNumbers():

    ## -- set role --

    pair = PARTICIPANT[1:] # set seed to last 2 digits of PARTICIPANTect ID [should be the same for both participants]
    if (PARTICIPANT[0] == '0'):
        role = 'A'
    elif (PARTICIPANT[0] == '1'):
        role = 'B'

    ## -- set random seed --

    seed_num = int(pair) # set seed to pair #

    ## -- set audio and networking settings --

    # default audio settings: mono, 16kHz sampling, 16bit encoding
    CHANNELS = 1
    RATE = 16000
    FORMAT = pyaudio.paInt16

    # UDP hole punch timeout (time for handshake), in seconds
    punchTimeout = 30

    # default port numbers
    # local
    portIn = 30002
    portOut = 30001
    # remote
    PortIn = 30002
    PortOut = 30001
    PortComm = 30003

    # set IP addresses
    if role == 'A': # needs IP for Prisma
        # IP = '192.168.0.103'
        IP = '10.9.154.26'
    elif role == 'B': # needs IP for Skyra
        # IP = '192.168.0.102'
        IP = '10.9.71.26'

    ## -- stimuli settings --

    exp_name = 'CONV'
    win_x = 1440 # 1920 # window size in px; for scanner comp: 1920
    win_y = 900 # 1080 # window size in px; for scanner comp: 1080
    win_fullscr = False
    stimuli_file = 'stimuli_sets.csv' # assumes this is in the same directory as this script
    read_file = 'read_text.csv' # assumes this is in the same directory as this script
    start_lag = 10  # time variable for start sync (seconds)
    items_n = 20 # number of prompts
    runs_n = 5 # number of runs
    letter_h = 0.06
    wrap_w = 1.4
    intro_time = 5 # in seconds
    trial_time = 60*3 # in seconds
    between_time = 5 # in seconds

    key_list_speak = ['1', '2', 'space', 'equal', 'escape'] # 1 to continue, 2 to pass the mic, = to start

    ## -- set default filenames --

    datet = datetime.datetime.now()
    date_str = datet.strftime('%Y%m%d_%H%M%S')
    savefileLog = _thisDir + os.sep + u'data/%s_%s_TimingsLog_%s.csv' % (exp_name, PARTICIPANT, date_str)

    return(punchTimeout, CHANNELS, RATE, FORMAT, date_str, savefileLog,
            portIn, portOut, PortIn, PortOut, PortComm, pair, role, seed_num, IP,
            exp_name, win_x, win_y, win_fullscr, stimuli_file, read_file, start_lag, items_n, runs_n, letter_h, wrap_w,
            intro_time, trial_time, between_time, key_list_speak)

## STUN query function
# This is called if you want to explore NAT properties.
# This is a wrapper around the stunclient tool. When STUN arg is set to 1,
# we just call it in a subprocess and capture some of its output.
# This function is not needed for the script to work, but it is
# included here for NAT diagnostic purposes.

def stunQuery():
    # list of a few servers to try
    serverList = [
                'stun.ekiga.net',
                'stun.ideasip.com',
                'stun.stunprotocol.org'
                ]

    # try servers until we get a successful test,
    # write output to terminal (NAT behavior type, internal and mapped
    # addressses, etc)
    stunFlag = False
    for i in serverList:
        print('Trying stun server ', i, '...')
        try:
            stunOutput = subprocess.check_output(
                        ['stunclient', '--mode', 'full', i],
                        universal_newlines=True)
            if bool(re.search('Behavior test: success', stunOutput)):
                for lines in stunOutput.splitlines():
                    print(lines)
                stunFlag = True
                break
        except:
            print('Stun query failed...')
    # if all queries fail, we have a problem
    if not stunFlag:
        print('All stun queries failed, no successful NAT behavior test')
    return stunFlag

## Socket function
# Opens simple UDP socket, binds it to given port number at localhost.

def openSocket(port):
    socketFlag = False
    # define socket
    try:
        socketUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('\nSocket created')
    except socket.error:
        print('\nFailed to create UDP socket')
    # bind port
    host = ''  # localhost
    try:
        socketUDP.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socketUDP.bind((host, port))
        print('\nUDP socket bound to local port: ', port)
        socketFlag = True
    except socket.error:
        print('\nFailed to bind UDP socket to local port ', port)
    return socketFlag, socketUDP

## Hole punch function
# Here, we assume that there is one side behind NAT. That side
# needs to initiate UDP hole punching (NAT traversal). This function
# handles UDP hole punching in eiher case, for both in- and outgoing
# communication

def punchThrough(NAT, socketIn, socketOut, socketComm, punchTimeout):

    global IP, PortIn, PortOut, PortComm
    # if other side is behind NAT, script just waits for connection,
    # both for socketIn and socketOut
    if NAT == 0:
        print('\n\nWaiting for other end to initiate handshake...\n')
        start = time.time()
        recFlag = False
        # for this part, socket is non-blocking with a timeout
        socketIn.settimeout(1)
        socketOut.settimeout(1)
        socketComm.settimeout(1)
        # until there is incoming message or max time is up
        while (not recFlag) & (abs(time.time()-start) < punchTimeout):
            try:
                incomingIn, addressIn = socketIn.recvfrom(1024)
                incomingOut, addressOut = socketOut.recvfrom(1024)
                incomingComm, addressComm = socketComm.recvfrom(1024)
                # if we have incoming message 'hello!'
                if (bool(incomingIn == 'hello!'.encode()) &
                    bool(incomingOut == 'hello!'.encode()) &
                    bool(incomingComm == 'hello!'.encode()) &
                        bool(addressIn[0] == addressOut[0])):
                        print('\nHandshake initiated from ', addressIn[0], ':',
                              addressIn[1], ' and :', addressOut[1])
                        IP = addressIn[0]

                        # NEW PART, SETTING REMOTE PORTS ACCORDING TO
                        # ADDRESS OF INCOMING PACKETS
                        PortOut = addressIn[1]
                        PortIn = addressOut[1]
                        PortComm = addressComm[1]
                        print('\nRemote ports are',
                              PortIn, PortOut, PortComm, '\n')

                        recFlag = True
            except:
                print('No handshake message yet...', end='\r', flush=True)
        # if time is over
        if not recFlag:
            IP = []
            return recFlag

        # send answer, wait until handshake is confirmed
        print('\nResponding...\n')
        recFlag = False
        start = time.time()
        # send answer and listen to next message, until time is up
        while (not recFlag) & (abs(time.time()-start) < punchTimeout):
            try:
                socketIn.sendto('hello!'.encode(), (IP, PortOut))
                socketOut.sendto('hello!'.encode(), (IP, PortIn))
                socketComm.sendto('hello!'.encode(), (IP, PortComm))
            except:
                print('\n\nCould not send "hello" packet to ', IP)
            # see if there was answer
            try:
                incomingIn, addressIn = socketIn.recvfrom(1024)
                incomingOut, addressOut = socketOut.recvfrom(1024)
                incomingComm, addressComm = socketComm.recvfrom(1024)
                if (bool(incomingIn == 'hello!'.encode()) &
                   bool(addressIn[0] == IP) &
                   bool(incomingOut == 'hello!'.encode()) &
                   bool(addressOut[0] == IP) &
                   bool(incomingComm == 'hello!'.encode()) &
                   bool(addressComm[0] == IP)):
                    print('\nHandshake confirmed, other end is ready\n')
                    recFlag = True
            except:
                print('No confirmation yet', end='\r', flush=True)
        # if there was no answer in the maximum allowed time
        if not recFlag:
            IP = []

    # if other end is behind NAT, it initiates hole punching. We assume
    # current side to be reachable via public IP and given port
    if NAT == 1:
        # actual handshake part
        print('\n\nInitiating handshake...\n')
        start = time.time()
        recFlag = False
        # for this part, socket is non-blocking with a timeout
        socketIn.settimeout(1)
        socketOut.settimeout(1)
        socketComm.settimeout(1)
        # send handshake and listen for answer until time is up
        # when sending, make sure to 'cross' the in and out ports between
        # local and remote
        print('\nSending handshake message "hi partner"...\n')
        while (not recFlag) & (abs(time.time()-start) < punchTimeout):
            try:
                socketIn.sendto('hi partner'.encode(), (IP, PortOut))
                socketOut.sendto('hi partner'.encode(), (IP, PortIn))
                socketComm.sendto('hi partner'.encode(), (IP, PortComm))
            except:
                print('\n\nCould not send "hi partner" packet to ', IP,)
            # see if there was an answer
            try:
                incomingIn, addressIn = socketIn.recvfrom(1024)
                incomingOut, addressOut = socketOut.recvfrom(1024)
                incomingComm, addressComm = socketComm.recvfrom(1024)
                if (bool(incomingIn == 'hi partner'.encode()) &
                   bool(addressIn[0] == IP) &
                   bool(incomingOut == 'hi partner'.encode()) &
                   bool(addressOut[0] == IP) &
                   bool(incomingComm == 'hi partner'.encode()) &
                   bool(addressComm[0] == IP)):
                    print('\nReceived answer, handshake confirmed\n')
                recFlag = True
            except:
                print('No proper answer yet', end='\r', flush=True)

        # if time is over
        if not recFlag:
            return recFlag

        # if handshake was successful, send a signal asking for audio
        if recFlag:
            # repeat final message a few times
            # 'cross' in and out ports across machines again
            for i in range(5):
                socketIn.sendto('please'.encode(),
                                (IP, PortOut))
                socketOut.sendto('please'.encode(),
                                 (IP, PortIn))
                socketComm.sendto('please'.encode(),
                                  (IP, PortComm))
            print('\nUDP hole punched, we are happy and shiny\n')

    # flush the sockets before we go on
    start = time.time()
    while abs(start-time.time()) < 1:
        try:
            incomingIn = socketIn.recv(1024)
            incomingOut = socketOut.recv(1024)
            incomingComm = socketComm.recv(1024)
        except:
            pass
    return recFlag

## Callback function for non-blocking pyaudio (portaudio) input
# Important!! We put everything that is to be done with the data into
# the callback function. Specifically, callback saves input audio,
# handles timestamps and counters and sends UDP packets.

# Expects output file to be open and ready for writing.
# Expects UDP socket and connection to server to be ready.
# Expects packetCounter to be set up.

def callbackInput(in_data, frame_count, time_info, status):
    # keep track of chunks
    global chunkCounter
    # refresh counter
    chunkCounter += 1
    # following line changed for python 2.7
    bytepacket = struct.pack('<l', chunkCounter)
    # write out new data before we mess with it
    fOut.write(in_data)
    # create bytearray from the audio chunk, so we can expand it
    dataArray = bytearray(in_data)
    # append data with timestamp and packetCounter
    timestamp = time.time()
    bytestamp = struct.pack('<d', timestamp)  # convert float into bytes
    # extend dataArray, get final packet to send
    dataArray.extend(bytepacket)
    dataArray.extend(bytestamp)
    in_data = bytes(dataArray)
    # send new data to other side
    try:
        socketOut.sendto(in_data, (IP, PortIn))
    except socket.error:
        print('Failed to send packet, chunkCounter = '+str(chunkCounter))
    # return data and flag
    return (in_data, pyaudio.paContinue)

## Callback function for non-blocking pyaudio (portaudio) output.

# In this version, we use a simple continuous buffer that collects
# all incoming packets on one end and is read out by the callback on the other.
# Important!!
# Expects output files to be open and ready for writing.
# Expects UDP socket and connection to server to be ready.
# Expects all four counters + changeFlag to be set up.
def callbackOutput(in_data, frame_count, time_info, status):
    global lastData, underFlowFlag
    # once the buffer is filled for the first time, startFlag is set and
    # callback can read from it
    if startFlag:
        # first check if there is enough data available to read
        if len(audioBuffer) > CHUNK*2:
            data = audioBuffer[0:CHUNK*2]
            del audioBuffer[0:CHUNK*2]
            lastData = data
        # if buffer is empty, update the underflow counter
        else:
            data = lastData
            underFlowFlag += 1

    # until startFlag is set, callback reads from a silence buffer (zeros)
    else:
        if len(silenceBuffer) > CHUNK*2:
            data = silenceBuffer[0:CHUNK*2]
            del silenceBuffer[0:CHUNK*2]
            lastData = data
        else:
            data = lastData

    data = bytes(data)
    fOut.write(data)
    return data, pyaudio.paContinue


## Function to set up mic
# Uses pyaudio (portaudio) for a non-blocking input device.
# Device is default input set on platform.

def micOpen(FORMAT, CHANNELS, RATE, CHUNK):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callbackInput,
                    start=False)  # IMPORTANT: don't start yet
    return stream, p

## Function to open output device
# Uses pyaudio (portaudio). Chooses default output device on platform.

def speakersOpen(FORMAT, CHANNELS, RATE, CHUNK):
    # open pyaudio (portaudio) device
    p = pyaudio.PyAudio()
    # open portaudio output stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callbackOutput,
                    start=False)  # IMPORTANT: don't start yet
    return stream, p

## Function to strip packetCounter and client timestamp from UDP packets
# the last 8 bytes is the timestamp, the 4 before that is the packetNumber
def packetParser(dataIn):
    dataArray = bytearray(dataIn)
    audio = dataArray[0:len(dataIn)-12]
    # using struct unpack to stay compatible with 2.7
    packetNumber = struct.unpack('<l', dataArray[len(dataIn)-12:len(dataIn)-8])
    packetNumber = packetNumber[0]
    timePacket = struct.unpack('<d', dataArray[len(dataIn)-8:len(dataIn)])
    return audio, packetNumber, timePacket

## Cleanup function for input (microphone)
# Close and terminate pyaudio, close socket, close files.
def cleanupInput():
    print('\nTransmission finished, cleaning up input...\n')
    # end signal in UDP packet
    for i in range(5):
        try:
            closePacket = b'thanks'
            socketOut.sendto(closePacket, (IP, PortIn))
            print('Sending closing packets', end='\r', flush=True)
        except:
            print('Sending closing packet failed')
    print('\nClosing portaudio input device, sockets, files\n')
    # pyaudio
    streamInput.stop_stream()
    streamInput.close()
    pIn.terminate()
    # sockets
    socketOut.close()
    socketComm.close()
    # files
    fOut.close()
    return

## Cleanup function for output (speakers)
# Close and terminate pyaudio, close socket, close files.
def cleanupOutput():
    print('\nTransmission finished, cleaning up output...\n')
    print('\nClosing portaudio output device, sockets, files\n')
    # pyaudio
    streamOutput.stop_stream()
    streamOutput.close()
    pOut.terminate()
    # sockets
    socketIn.close()
    return

## Wav creator function for the binary recorded data
def wavmaker(filename, CHANNELS, RATE):
    # create wav for audio
    f = open(filename, 'rb')
    audio = f.read()
    wavef = wave.open(filename+'.wav', 'w')
    wavef.setnchannels(CHANNELS)
    wavef.setsampwidth(2)
    wavef.setframerate(RATE)
    wavef.writeframes(audio)
    wavef.close()
    f.close()
    return

## Cleanup on keypress ESC
def cleanup(queueInput, queueOutput, audioInput, audioOutput, fLog):
    print('\nTerminating...\n')
    # killing audio I/O
    queueInput.put('die')
    queueOutput.put('die')
    audioInput.join()
    audioOutput.join()

    # close log files
    fLog.close()

    try:
        # create wav files
        wavmaker(savefileOut, CHANNELS, RATE)
    except:
        print('\nNo audio file was created, so unable to turn file to .wav format')

    try:
        # save csv (these shouldn't be strictly necessary, should auto-save)
        thisExp.saveAsWideText(filename+'.csv')
        thisExp.saveAsPickle(filename)
        logging.flush()
    except:
        print('\nExperiment did not start so no csv can be saved')

    sys.exit()

## Function to open all needed sockets and handle NAT

def networkInit(STUN, NAT, portIn, portOut, punchTimeout):
    global socketIn, socketOut, socketComm

    # if STUN was asked
    if STUN:
        stunFlag = stunQuery()
        if not stunFlag:
            print('\n\nSTUN query failed, something is wrong. Check ' +
                  'connection. Do you have stuntman installed?')
    # UPD sockets for transmission
    socketFlag, socketOut = openSocket(portOut)
    if not socketFlag:
        print('\n\nCould not create or bind UDP socket. Uh-oh.')
        sys.exit()
    socketFlag, socketIn = openSocket(portIn)
    if not socketFlag:
        print('\n\nCould not create or bind UDP socket. Uh-oh.')
        sys.exit()
    socketFlag, socketComm = openSocket(PortComm)
    if not socketFlag:
        print('\n\nCould not create or bind UDP socket. Uh-oh.')
        sys.exit()
    # Hole punch
    recFlag = punchThrough(NAT, socketIn, socketOut, socketComm, punchTimeout)
    if not recFlag:
        print('\n\nSomething went wrong at NAT traversal. Uh-oh.')
        sys.exit()
    # set sockets to non-blocking
    socketOut.settimeout(0)
    socketIn.settimeout(0.1)
    socketComm.settimeout(0.1)
    return

## Main input (microphone) function. Handles audio input and
# transmission. Should be called in separate process (multiprocessing()),
# after networkInit(), at the same time as outputProcess()
def inputProcess(FORMAT, CHANNELS, RATE, CHUNK, queueInput):

    global chunkCounter, streamInput, pIn
    # init chunkCounter
    chunkCounter = 0
    # open input dev
    streamInput, pIn = micOpen(FORMAT, CHANNELS, RATE, CHUNK)

    # print start message
    print('\nEverything seems all right, channel open on our side.\n')

    while True:
        note = queueInput.get()
        if note == "stop":
            streamInput.stop_stream()
        elif note == "start":
            # start input stream
            start = time.time()
            streamInput.start_stream()
        elif note == "die":
            break
        # wait until all audio is sent
        while streamInput.is_active():
            time.sleep(0.01)
            # if escape key was pressed, terminate
            if not queueInput.empty():
                break

    # input cleanup
    cleanupInput()

    return

## Main output (receiver) function. Handles audio output and
# packet control. Should be called in separate process (multiprocessing()),
# after networkInit(), at the same time as inputProcess()
def outputProcess(BUFFER, CHUNK, FORMAT, CHANNELS, RATE, queueOutput):

    # these need to be global...
    global underFlowFlag, startFlag, audioBuffer, streamOutput
    global pOut, silenceBuffer, lastData

    # initialize buffer underflow / overflow flags, callback start flag
    underFlowFlag = 0
    startFlag = 0
    overFlowFlag = 0

    # Lists to store incoming packet numbers, client side timestamps and
    # server side timestamps
    packetListClient = list()
    packetListClient.append(0)
    timeListClient = list()
    timeListServer = list()

    # open output dev
    streamOutput, pOut = speakersOpen(FORMAT, CHANNELS, RATE, CHUNK)
    print('\nAudio output set up, waiting for transmission.')

    # counter for all received UDP packets
    packetCounter = 0

    # create buffer for incoming packets
    audioBuffer = bytearray()

    # start stream with a silent buffer (silence buffer)
    silenceBuffer = b'x\00'*2*CHUNK*BUFFER
    silenceBuffer = bytearray(silenceBuffer)
    lastData = silenceBuffer[0:CHUNK*2]

    while True:
        note = queueOutput.get()
        if note == "stop":
            streamOutput.stop_stream()
        elif note == "start":
            streamOutput.start_stream()
        elif note == "die":
            break
        # wait until all audio is sent
        while streamOutput.is_active():

            # if escape key was pressed, terminate
            if not queueOutput.empty():
                break

            # receive UDP packet - remember this is in non-blocking mode now!
            packet = []
            try:
                packet = socketIn.recv(CHUNK*4)
            except:
                pass
            # if we received anything
            if packet:
                # other end can end session by sending a specific message
                # ('thanks')
                if packet == b'thanks':
                    print('thanks (end message) received, finishing output')
                    break
                # parse packet into data and the rest
                data, packetNumber, timePacket = packetParser(packet)
                # adjust packet counter
                packetCounter += 1

                # do a swap if packetNumber is smaller than last
                if packetNumber > 3:
                    if packetNumber < packetListClient[-1]:
                        try:  # buffer could be empty...
                            audioBuffer.extend(audioBuffer[-CHUNK*2:])
                            audioBuffer[-CHUNK*4:-CHUNK*2] = data
                        except:
                            audioBuffer.extend(data)
                    else:
                        # otherwise just append audioBuffer with new data
                        audioBuffer.extend(data)
                else:
                    # otherwise just append audioBuffer with new data
                    audioBuffer.extend(data)

                # get server-side timestamp right after writing data to buffer
                timeListServer.append(time.time())

                # set startFlag for callback once buffer is filled for the first
                # time
                if packetCounter == BUFFER:
                    startFlag = 1

                # if audioBuffer is getting way too long, chop it back, the
                # threshold is two times the normal size
                if len(audioBuffer) > 2*CHUNK*BUFFER*2:
                    del audioBuffer[0:2*CHUNK*BUFFER]
                    overFlowFlag += 1

                # append timePacket and packetNumber lists
                packetListClient.append(packetNumber)
                timeListClient.append(float(timePacket[0]))

    # end messages
    messagesOutput(run_n, packetCounter, timeListServer, timeListClient,
                   packetListClient, overFlowFlag)
    # cleanup
    cleanupOutput()

    return

#%% Function to display closing stats and messages for output
def messagesOutput(run_n, packetCounter, timeListServer, timeListClient, packetListClient, overFlowFlag):

    try:
        # summary message
        print('\n\n'+'Time taken for all chunks: ' +
              str(timeListServer[-1]-timeListServer[0]) + ' = ' + str((timeListServer[-1]-timeListServer[0])/60) + ' minutes')
        # more diagnostic messages
        print('\nReceived '+str(packetCounter)+' audio chunks')
        # underflow events
        print('\nBuffer underflow occurred '+str(underFlowFlag)+' times')
        # overflow events
        print('\nBuffer overflow occurred '+str(overFlowFlag)+' times')
        # print average transmission time
        timeListDiff = list(map(sub, timeListServer, timeListClient))
        print('\nAverage difference between client and server side timestamps: ',
              sum(timeListDiff) / len(timeListDiff), ' secs \n\nClient timestamp '
              'is taken after reading audio input buffer \nServer timestamp is '
              'taken when pushing the received data into audio output buffer\n\n')
    except:
        print('\nCould not provide summary message')

## Function to integrate the pieces and run the whole thing
def goGo(NAT, STUN, LOGTTL, PARTICIPANT, RUN):

    ## -- take care of network, audio, and TTL recording --

    # these need to be global for callbacks, etc.
    global IP, fOut, PortIn, PortOut, PortComm, run_n

    run_n = RUN

    # load all settings, magic numbers
    [punchTimeout, CHANNELS, RATE, FORMAT, date_str, savefileLog,
            portIn, portOut, PortIn, PortOut, PortComm, pair, role, seed_num, IP,
            exp_name, win_x, win_y, win_fullscr, stimuli_file, read_file,
            start_lag, items_n, runs_n, letter_h, wrap_w,
            intro_time, trial_time, between_time, key_list_speak] = magicNumbers()

    # networkInit
    networkInit(STUN, NAT, portIn, portOut, punchTimeout)

    # open files
    fLog = open(savefileLog, 'w')  # text file
    # set filenames
    savefileOut = _thisDir + os.sep + u'data/CONV_' + PARTICIPANT + '_RecordedAudio_' + date_str

    # open files we will use for writing stuff out
    fOut = open(savefileOut, 'wb') # audio file

    # write headers for text files
    fLog.write('run,trial,item,presentation,role,time,audio_position\n')

    # audio I/O processes and TTL recording run in separate processes
    queueInput = multiprocessing.Queue()
    queueOutput = multiprocessing.Queue()
    audioInput = multiprocessing.Process(name='audioInput',
                                         target=inputProcess,
                                         args=(FORMAT,
                                               CHANNELS,
                                               RATE,
                                               CHUNK,
                                               queueInput,))
    audioOutput = multiprocessing.Process(name='audioOutput',
                                          target=outputProcess,
                                          args=(BUFFER,
                                                CHUNK,
                                                FORMAT,
                                                CHANNELS,
                                                RATE,
                                                queueOutput))
    audioInput.start()
    audioOutput.start()

    ## -- check audio status --
    if not audioInput.is_alive() or not audioOutput.is_alive():
        print('\nAudio inputs and outputs are not alive. Terminating so you can start over again.')
        cleanup(queueInput, queueOutput, audioInput,
                   audioOutput, fLog)

    ## -- import the rest of PsychoPy --

    # If psychopy is imported before the multiprocesses start, the code won't work
    from psychopy import locale_setup, sound, visual, core, data, event, logging, clock

    ## -- set up save files --

    # Data file name stem = absolute path + name; later add .csv, .log, etc
    filename = _thisDir + os.sep + u'data/%s_%s_%s' % (exp_name, PARTICIPANT, date_str)

    # An ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(name=exp_name, version='',
        extraInfo='', runtimeInfo=None,
        originPath= '_thisDir' + os.sep + u'CONV_read.py',
        savePickle=True, saveWideText=True,
        dataFileName=filename)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

    ## -- import stimuli csv --

    stim_df = pd.read_csv(stimuli_file)
    read_df = pd.read_csv(read_file)

    ## -- set up things that need to be randomized / shuffled --

    # set random seed
    random.seed(seed_num)

    # shuffle list of first speaker order
    list_first_speaker = [random.sample(['A','B'], 2), random.sample(['A','B'], 2), \
    random.sample(['A','B'], 2), random.sample(['A','B'], 2), \
    random.sample(['A','B'], 2), random.sample(['A','B'], 2), \
    random.sample(['A','B'], 2), random.sample(['A','B'], 2), \
    random.sample(['A','B'], 2), random.sample(['A','B'], 2)]
    list_first_speaker = [item for sublist in list_first_speaker for item in sublist]

    # get random order of prompts for each set
    prompts_item_order = [random.sample([1,2], 2), random.sample([3,4], 2), \
    random.sample([5,6], 2), random.sample([7,8], 2), \
    random.sample([9,10], 2), random.sample([11,12], 2), \
    random.sample([13,14], 2), random.sample([15,16], 2), \
    random.sample([17,18], 2), random.sample([19,20], 2)]
    prompts_item_order = [item for sublist in prompts_item_order for item in sublist]
    prompts_item_idx = [float(x) - 1 for x in prompts_item_order]
    # counterbalance order of whole vs. split presentation
    if int(pair) % 2 == 0:
        prompts_present_order = ['whole']*(items_n // 2) + ['split']* (items_n // 2)
    else:
        prompts_present_order = ['split']*(items_n // 2) + ['whole']* (items_n // 2)
    prompts_text_order = stim_df['Question'].iloc[prompts_item_idx].tolist()
    sets_order = stim_df['Set'].iloc[prompts_item_idx].tolist()

    # grab text for read presentation
    read_turns = []
    for idx in range(items_n):
        a = read_df["Text"].loc[read_df["Item"] == idx + 1]
        b = []
        [b.append(x) for x in a]
        read_turns.append(b)

    ## -- start visual part of the experiment --

    try:
        win = visual.Window(size=[win_x, win_y],
                            color='black',
                            fullscr=win_fullscr,
                            screen=0,
                            )
        win.mouseVisible = False
    except:
        print('\nProblem while setting up window')
        cleanup(queueInput, queueOutput, audioInput,
                   audioOutput, fLog)

    for run_n in range(RUN, runs_n + 1):

        # slice list based on run
        num_items_per_run = items_n // runs_n
        prompts_item_run = prompts_item_order[num_items_per_run*run_n-num_items_per_run:num_items_per_run*run_n]
        prompts_text_run = prompts_text_order[num_items_per_run*run_n-num_items_per_run:num_items_per_run*run_n]
        prompts_present_run = prompts_present_order[num_items_per_run*run_n-num_items_per_run:num_items_per_run*run_n]
        list_first_speaker_run = list_first_speaker[num_items_per_run*run_n-num_items_per_run:num_items_per_run*run_n]
        sets_run = sets_order[num_items_per_run*run_n-num_items_per_run:num_items_per_run*run_n]

        ## -- set up TrialHandler --

        # create list of stimuli
        stim_list = []
        for idx in range(num_items_per_run):
            # append a python 'dictionary' to the list
            stim_list.append({'run': run_n, 'set': sets_run[idx], 'index': idx, 'trial': idx + 1, 'item': prompts_item_run[idx], 'question': prompts_text_run[idx], 'presentation': prompts_present_run[idx], 'first_speaker': list_first_speaker_run[idx]})

        # organize them with the trial handler
        trials = data.TrialHandler(stim_list, 1, method = 'sequential',
        extraInfo={'PARTICIPANT': PARTICIPANT, 'role': role, 'date': date_str})

        instr_text = ('Task:\n\nYou will take turns reading lines from conversations on ' +
                        str(items_n // runs_n) + ' different topics.\n\n' +
                       '- When it is your turn to read, your partner will listen.\n'
                       '- When it is your partner\'s turn to read, you will listen.\n'
                       '\n\nA timer will show you how many seconds are left for the entire conversation on a given topic. ' +
                       'You will have ' + '{0:.0f}'.format(trial_time/60) + ' minutes for each topic.\n\n'
                       'Please try your best to not rush through the lines. We want you to sound as if you were *actually* ' +
                       'conversing with another person.\n\nPress the space bar when you are ready to begin.'
                       '\n\nRun ' + str(run_n) + ' out of ' + str(runs_n))

        # Instructions: basic text instructions, two pages
        instructions = visual.TextStim(win,name='instructions',
                                        text=instr_text,
                                        color='white',
                                        height=letter_h,
                                        pos=(0, 0),
                                        wrapWidth=1.2)
        # draw, flip
        instructions.draw()
        win.flip()

        # wait for trigger to start task
        triggered = False
        while not triggered:
            keys = event.getKeys(key_list_speak)
            if keys:
                print('keys pressed: ' + str(keys))
                # if event.getKeys returns a '=', its a TTL
                if keys[0] == 'space':
                    triggered = True
                # escape quits
                elif keys[0] == 'escape':
                    cleanup(queueInput, queueOutput, audioInput, audioOutput, fLog)
                    return

        ## -- sync computers --

        # put up a sync screen while we are waiting for startTimeCommon
        syncText = 'Syncing start time with your partner...'
        sync_stim = visual.TextStim(win,name='sync_stim',
                                    text=syncText,
                                    color='white',
                                    height=letter_h,
                                    wrapWidth=wrap_w)
        sync_stim.draw()
        win.flip()

        # Sync process: (1) handshake to start, (2) exchange time stamps,
        # derive common start time (max of time stamps + start_lag)
        commFlag = True
        incoming = []
        # first a handshake for sync
        print('\nStarting sync\n')
        while commFlag:
            # send packets
            try:
                socketComm.sendto('syncTimeNow'.encode(), (IP, PortComm))
            except:
                print('\nProblem sending a syncTimeNow packet...\n')
                pass
            try:
                incoming = socketComm.recv(CHUNK)
            except:
                pass
            if incoming == 'syncTimeNow'.encode():
                incoming = []
                # time stamp on our side
                timeHere = time.time()
                print('\nReceived sync handshake, sending timeHere',
                      str(timeHere), '\n')
                while True:
                    # escape quits
                    if event.getKeys(keyList=['escape']):
                        cleanup(queueInput, queueOutput, audioInput, audioOutput,
                                   fLog)
                        return
                    # send our time stamp
                    for i in range(2):
                        try:
                            socketComm.sendto(struct.pack('<d', timeHere),
                                              (IP, PortComm))
                        except:
                            print('\nProblem sending a timeHere packet...\n')
                            pass
                    # read out socket
                    try:
                        incoming = socketComm.recv(CHUNK)
                    except:
                        pass
                    # if readout data is what we would expect, create startTime
                    if bool(incoming) & bool(len(incoming) == 8):
                        print('\nGot incoming time\n')
                        # unpack time stamp from other side
                        timeThere = struct.unpack('<d', incoming)[0]
                        print('\nIncoming timeThere is',
                              str(timeThere), '\n')
                        # start is at the max of the two timestamps + a predefined lag
                        startTimeCommon = max(timeThere, timeHere) + start_lag
                        print('\nGot shared startTimeCommon:',
                              str(startTimeCommon), '\n')
                        commFlag = False
                        # insurance policy - send it last time
                        for i in range(2):
                            socketComm.sendto(struct.pack('<d', timeHere),
                                              (IP, PortComm))
                        break

        # common start is synced at a precision of
        # keyboard polling (few ms) + ntp diff + hardware jitter
        while time.time() < startTimeCommon:
            if event.getKeys(keyList=['escape']):
                cleanup(queueInput, queueOutput, audioInput, audioOutput, fLog)
                return

        # start logs
        # log audio file object positions at start
        # hdr: fLog.write('run,trial,item,presentation,role,time,audio_position\n')
        fLog.write(str(run_n) + ',,,,,' + str(startTimeCommon) + ',' + str(fOut.tell()) + '\n')

        # mute the mic and speaker
        queueInput.put('stop')
        queueOutput.put('stop')

        # flip screen
        win.flip()
        # track time
        global_clock = core.Clock()

        # blank for 10 seconds
        while global_clock.getTime() <  10:
            keys = event.getKeys(key_list_speak)
            if keys:
                # escape quits
                if keys[0] == 'escape':
                    cleanup(queueInput, queueOutput, audioInput, audioOutput, fLog)
                    return

        ## -- start loop --

        # initialize
        trial_listen_text = 'YOUR TURN TO LISTEN'
        trial_read_text = 'YOUR TURN TO READ'
        trial_turn_next = '[Press "1" to continue reading.]'
        trial_turn_last = '[This is the end of your turn. Press "2" to pass the mic.]'
        trial_timer_text = str(trial_time) + ' seconds'
        trial_prompt_text = ''
        trial_role_text = ''
        line_show_text = ''
        end_text = 'You have reached the end of the experiment. Please refer to the experimenter for further instructions.'

        trial_timer = visual.TextStim(win=win, name = 'trial_timer', text = trial_timer_text,
            pos=(0, -0.6), height=letter_h, color='white')
        trial_prompt = visual.TextStim(win=win, name = 'trial_prompt', text = trial_prompt_text,
            pos=(0, 0.2), height=letter_h, color='white')
        trial_role = visual.TextStim(win=win, name = 'trial_role', text = trial_role_text,
            pos=(0, 0.6), height=letter_h, wrapWidth=wrap_w, color='white')
        line_show = visual.TextStim(win=win, name = 'trial_line', text = line_show_text,
            pos=(0, 0), height=letter_h, wrapWidth=wrap_w, color='white')
        end = visual.TextStim(win=win, name = 'end', text = end_text,
            pos=(0, 0), height=letter_h, wrapWidth=wrap_w, color='white')

        for trial_i, this_trial in enumerate(trials):

            print('\n\n' + str(this_trial))

            # set trial-related clocks
            trial_clock_timer = core.CountdownTimer() # to track time remaining of each (non-slip) routine
            trial_clock = core.Clock() # to track time related to key responses

            ## -- set up texts displayed to participants --

            if ((this_trial['first_speaker'] == 'A') & (role == 'A')) or ((this_trial['first_speaker'] == 'B') & (role == 'B')):
                turn_role = 'speaker'
                trial_role_text = trial_read_text
                trial_prompt_text = '\nPrompt # ' + str(this_trial['trial']) + ':\n\n' + str(this_trial['question'])
            else:
                turn_role = 'listener'
                trial_role_text = trial_listen_text
                trial_prompt_text = 'Prompt # ' + str(this_trial['trial']) + ':\n\n' + 'Your partner received the prompt and will start talking shortly.'

            # update screen timer
            trial_role.setText(trial_role_text, log=False)
            trial_prompt.setText(trial_prompt_text, log=False)

            ## -- display trial intro --

            trial_prompt.setAutoDraw(True)
            win.flip()
            trial_clock_timer.add(intro_time)

            # log audio file object position for each trial
            # hdr: fLog.write('run,trial,item,presentation,role,time,audio_position\n')
            fLog.write(str(run_n) + ',' + str(this_trial['trial']) + ',' + str(this_trial['item']) + ',' + str(this_trial['presentation']) + ',trial_intro,' + str(time.time()) + ',' + str(fOut.tell()) + '\n')

            while trial_clock_timer.getTime() > 0:
                keys = event.getKeys(key_list_speak)
                if keys:
                    # escape quits
                    if keys[0] == 'escape':
                        cleanup(queueInput, queueOutput, audioInput, audioOutput, fLog)
                        return

            ## -- display trial --

            # drawing attributes
            trial_prompt.setAutoDraw(False)
            trial_timer.setAutoDraw(True)
            trial_role.setAutoDraw(True)
            line_show.setAutoDraw(True)
            win.flip()

            # update
            trial_onset = global_clock.getTime()
            trial_key = event.BuilderKeyResponse()
            trial_clock.reset()
            trial_clock_timer.add(trial_time)
            rt_list = []

            # if speaker, start mic stream and stop speaker stream, if listener, vice versa
            if turn_role == 'speaker':
                queueInput.put('start')
                queueOutput.put('stop')
            elif turn_role == 'listener':
                queueInput.put('stop')
                queueOutput.put('start')

            # log audio file pointer positions at start of trial
            # hdr: fLog.write('run,trial,item,presentation,role,time,audio_position\n')
            fLog.write(str(run_n) + ',' + str(this_trial['trial']) + ',' + str(this_trial['item']) + ',' +
            str(this_trial['presentation']) + ',' +  str(turn_role) + ',' + str(timeHere) + ',' + str(fOut.tell()) + '\n')

            # get all the turns associated with the trial
            trial_text = read_turns[this_trial['item'] - 1]
            num_turns = len(trial_text)

            for turn_i, turn in enumerate(trial_text):

                # switch roles for each turn after the first turn
                if turn_i > 0:
                    if turn_role == 'listener':
                        turn_role = 'speaker'
                        trial_role_text = trial_read_text
                        # start the mic
                        queueInput.put('start')
                        queueOutput.put('stop')
                    elif turn_role == 'speaker':
                        turn_role = 'listener'
                        trial_role_text = trial_listen_text
                        # stop the mic
                        queueInput.put('stop')
                        queueOutput.put('start')
                trial_role.setText(trial_role_text, log=False)
                # log audio file pointer positions at turn changes
                # hdr: fLog.write('run,trial,item,presentation,role,time,audio_position\n')
                fLog.write(str(run_n) + ',' + str(this_trial['trial']) + ',' + str(this_trial['item']) + ',' + str(this_trial['presentation']) + ',' +  'speaker' + ',' + str(timeHere) + ',' + str(fOut.tell()) + '\n')

                text_whole = turn
                text_split = text_whole.split('. ')
                if this_trial['presentation'] == 'whole':
                    text_show = [text_whole]
                elif this_trial['presentation'] == 'split':
                    text_show = text_split

                num_lines = len(text_show)

                for line_i, line in enumerate(text_show):

                    if turn_role == 'speaker':
                        if line_i < (num_lines - 1):
                            if line[-1] in string.punctuation:
                                line_show_text = line + '\n\n' + trial_turn_next
                            else:
                                line_show_text = line + '.\n\n' + trial_turn_next
                        else:
                            if line[-1] in string.punctuation:
                                line_show_text = line + '\n\n' + trial_turn_last
                            else:
                                line_show_text = line + '.\n\n' + trial_turn_last
                    elif turn_role == 'listener':
                        line_show_text = ''

                    line_show.setText(line_show_text, log=False)

                    while trial_clock_timer.getTime() > 0:
                        keys = event.getKeys(key_list_speak)
                        if keys:
                            if keys[0] == '1':
                                if (turn_role == 'speaker') and (line_i < num_lines - 1):
                                    # transmit turn info to other computer
                                    # send packets
                                    try:
                                        socketComm.sendto('1'.encode(), (IP, PortComm))
                                        timeHere = time.time()
                                        print('\nTime when the keypress was sent out: ' + str(datetime.datetime.fromtimestamp(timeHere)))
                                    except:
                                        print('\nProblem sending a keypress packet...\n')
                                        pass
                                    break
                            elif keys[0] == '2':
                                # if this participant is the speaker and he/she passes the mic - switch roles
                                # if participant is the listener and the speaker presses key,
                                # it is now the participant's turn to speak
                                if turn_role == 'speaker' and (line_i == num_lines - 1):
                                    rt_list.append(trial_clock.getTime())
                                    trial_key.keys = keys[-1]  # just the last key pressed
                                    # transmit turn info to other computer
                                    # send packets
                                    try:
                                        socketComm.sendto('2'.encode(), (IP, PortComm))
                                        timeHere = time.time()
                                        print('\nTime when the keypress was sent out: ' + str(datetime.datetime.fromtimestamp(timeHere)))
                                    except:
                                        print('\nProblem sending a keypress packet...\n')
                                        pass
                                    break
                            # escape quits
                            elif keys[0] == 'escape':
                                # transmit turn info to other computer
                                # send packets
                                try:
                                    socketComm.sendto('esc'.encode(), (IP, PortComm))
                                    timeHere = time.time()
                                    print('\nTime when the keypress was sent out: ' + str(datetime.datetime.fromtimestamp(timeHere)))
                                except:
                                    print('\nProblem sending a keypress packet...\n')
                                    pass
                                cleanup(queueInput, queueOutput, audioInput, audioOutput, fLog)
                                return

                        incoming = []
                        # if participant is a listener
                        if turn_role == "listener":
                            speaker =  list(set(['A','B'])-set([role]))[0]
                            try:
                                incoming = socketComm.recv(CHUNK)
                            except:
                                pass
                            if incoming == '1'.encode():
                                break
                            if incoming == '2'.encode():
                                trial_key.keys = '2'  # just the last key pressed
                                rt_list.append(trial_clock.getTime())
                                # time stamp on our side
                                print('\nReceived keypress at time ',
                                      str(datetime.datetime.fromtimestamp(time.time())))
                                break
                            if incoming == 'esc'.encode():
                                cleanup(queueInput, queueOutput, audioInput, audioOutput, fLog)
                                return

                        # update screen timer
                        trial_timer.setText(str(round(trial_clock_timer.getTime())) + ' seconds', log=False)

                        # refresh the screen
                        win.flip()

            ## -- blank screen --
            trial_prompt.setAutoDraw(False)
            trial_timer.setAutoDraw(False)
            trial_role.setAutoDraw(False)
            line_show.setAutoDraw(False)
            win.flip()

            queueInput.put('stop')
            queueOutput.put('stop')
            # log audio file pointer positions at start of trial
            # hdr: fLog.write('run,trial,item,presentation,role,time,audio_position\n')
            fLog.write(str(run_n) + ',' + str(this_trial['trial']) + ',' + str(this_trial['item']) + ',' + str(this_trial['presentation']) + ',trial_end,' + str(timeHere) + ',' + str(fOut.tell()) + '\n')

            # duration between trials
            trial_clock_timer.add(between_time)
            while trial_clock_timer.getTime() > 0:
                keys = event.getKeys(key_list_speak)
                if keys:
                    # escape quits
                    if keys[0] == 'escape':
                        cleanup(queueInput, queueOutput, audioInput, audioOutput, fLog)
                        return

            ## -- save responses --
            thisExp.addData('run', this_trial['run'])
            thisExp.addData('set', this_trial['set'])
            thisExp.addData('trial', this_trial['trial'])
            thisExp.addData('item', this_trial['item'])
            thisExp.addData('presentation', this_trial['presentation'])
            thisExp.addData('first_speaker', this_trial['first_speaker'])
            thisExp.addData('onset', trial_onset)
            if trial_key.keys in ['', [], None]:  # No response was made
                trial_key.keys=None
                thisExp.nextEntry()
            trials.addData('keys',trial_key.keys)
            if trial_key.keys != None:  # we had a response
                thisExp.addData('rt', rt_list)
                thisExp.nextEntry()

            # in case trial is not non-slip safe. resetting non-slip timer
            trial_clock.reset()

    ## -- end --

    time_end = global_clock.getTime()

    # close audio
    queueInput.put('die')
    queueOutput.put('die')
    audioInput.join()
    audioOutput.join()
    fOut.close()

    # create .wav files
    wavmaker(savefileOut, CHANNELS, RATE)

    # save csv (these shouldn't be strictly necessary, should auto-save)
    thisExp.saveAsWideText(filename+'.csv')
    thisExp.saveAsPickle(filename)
    logging.flush()
    # close log files + psychopy stuff
    fLog.close()

    # show end screen
    end.setAutoDraw(True)
    win.flip()

    # Total duration
    total_duration = global_clock.getTime()
    print('Total duration: ' + str(total_duration / 60) + ' minutes')

    while True:
        keys = event.getKeys(key_list_speak)
        if keys:
            # equal quits
            if keys[0] == 'equal':
                break

    cleanup(queueInput, queueOutput, audioInput, audioOutput, fLog)

    win.close()
    # make sure everything is closed down or data files will save again on exit
    thisExp.abort()
    core.quit()

    return


#%% MAIN
if __name__ == '__main__':

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--PARTICIPANT',
        nargs='?',
        type=str,
        required=True,
        default='',
        help='Specify participant #')
    parser.add_argument(
        '-r',
        '--RUN',
        nargs='?',
        type=int,
        required=True,
        help='Enter run number')
    parser.add_argument(
        '-n',
        '--NAT',
        nargs='?',
        type=int,
        default=1,
        help='Flag for local NAT: set to 0 if ports are forwarded through ' +
             'NAT, set to 1 otherwise. If 1, provide IP as well! Default = 1')
    parser.add_argument(
        '-c',
        '--CHUNK',
        nargs='?',
        type=int,
        default=512,
        help='Audio chunk (packet) size in frames (1 frame = 2 bytes with ' +
             'current format settings). Integer. Default = 512')
    parser.add_argument(
        '-b',
        '--BUFFER',
        nargs='?',
        type=int,
        default=4,
        help='No. of chunks to buffer for audio output. Integer. Default = 4')
    parser.add_argument(
        '-s',
        '--STUN',
        nargs='?',
        type=int,
        default=0,
        help='Flag to run stunclient (1) or not (0) ' +
             'at the beginning of the script. Requires installed ' +
             'stunclient. Default = 0')
    parser.add_argument(
        '-l',
        '--LOGTTL',
        nargs='?',
        type=int,
        default=0,
        help='Flag for logging scanner ttl signals and their timestamps: ' +
             '0 means no ttl log (for testing outside scanner), 1 means ' +
             'ttl log. Default = 1')
    args = parser.parse_args()

    # check inputs
    if 0 <= args.NAT <= 1:
        pass
    else:
        raise ValueError('Unexpected value for argument NAT ',
                         '(should be 0 or 1)')
    # the following check for power of two is a really cool trick!
    if ((args.CHUNK != 0) and ((args.CHUNK & (args.CHUNK - 1)) == 0) and
       (not (args.CHUNK < 128)) and (not (args.CHUNK > 4096))):
        pass
    else:
        raise ValueError('CHUNK should be power of two, between 128 and 4096.')
    if 1 <= args.BUFFER <= 25:
        pass
    else:
        raise ValueError('Unexpected value for argument BUFFER. ',
                         '(please have it 1 <= and <= 25.')
    if 0 <= args.STUN <= 1:
        pass
    else:
        raise ValueError('Unexpected value for argument STUN ',
                         '(should be 0 or 1)')
    if 1 <= args.RUN <= 5:
        pass
    else:
        raise ValueError('Unexpected value for argument RUN ',
                         '(should be between 1 and 6)')

    global BUFFER, CHUNK, PARTICIPANT, RUN
    BUFFER = args.BUFFER
    CHUNK = args.CHUNK
    PARTICIPANT = args.PARTICIPANT
    RUN = args.RUN

    # Run experiment (function goGo)
    goGo(args.NAT,
         args.STUN,
         args.LOGTTL,
         args.PARTICIPANT,
         args.RUN)
    # End
    print('\n\nEverything ended / closed the way we expected. Goodbye!\n\n')
