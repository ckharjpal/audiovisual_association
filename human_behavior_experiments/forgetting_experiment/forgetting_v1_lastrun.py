#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.1.2),
    on August 28, 2023, at 13:12
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.1.2'
expName = 'forgetting_v1'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': ''}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='F:\\Downloads\\LEAP lab\\behavioral_experiment_2023\\version_1\\forgetting_v1_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1536, 864], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# Setup ioHub
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# Initialize components for Routine "welcome_screen"
welcome_screenClock = core.Clock()
welcome_text = visual.TextStim(win=win, name='welcome_text',
    text='Hi :)\n\nWe are glad that you could spare some time to participate in our experiment. You will be asked learn the audio and object pair presented and then match them in testing. Please ensure that you are seated comfortably in front of your screen.\n\nIf you have any questions about this study you may contact Chandrakant Harjpal(hchandrakant@iisc.ac.in) ,Prof. SP Arun (sparun @ iisc.ac.in), Centre for Neuroscience or Prof. Sriram Ganapathy(sriramg@iisc.ac.in), Electrical Engineering Department, IISc. \n\nYour participation in this study is voluntary. You may stop the task at any time if you do not wish to proceed. Your decision to withdraw will not affect your relations with your institute/university in any way. You are encouraged to ask questions about this study at any time. All your information and data collected will be kept completely confidential. No reference will be made in written or oral materials that could link you to this study. \n\n\nPARTICIPANT CONSENT\nI understand what is required of me, and all my questions have been answered. I now hereby give my consent of my own free will to participate in this experiment. \n\nPress SPACE to continue, else\nPress ESCAPE to exit',
    font='Open Sans',
    pos=(0, 0), height=0.027, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
welcome_resp = keyboard.Keyboard()

# Initialize components for Routine "AudioCheck"
AudioCheckClock = core.Clock()
audio_check_text = visual.TextStim(win=win, name='audio_check_text',
    text='Please ensure that your speakers are working and are set at a comfortable volume. \n\nPress SPACE if you were able to hear the sound at a comfortable volume. \n\nIf not, modify the volume of your speaker. Press R to hear it again. Press SPACE when you are comfortable with the volume. ',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
AudioResp = keyboard.Keyboard()
audio_check_sound = sound.Sound('audios/kinch_0.wav', secs=1.0, stereo=True, hamming=True,
    name='audio_check_sound')
audio_check_sound.setVolume(1.0)

# Initialize components for Routine "practice_info"
practice_infoClock = core.Clock()
practice_ins_text = visual.TextStim(win=win, name='practice_ins_text',
    text="Practice block\n\nHere, you will have to learn only two object-spoken word pairs. The image and audio will play simultaneously, you can look at image as long as you want and replay audio as many times as you want but can't go back.\n\nPress Space to start practice block",
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
practice_ins_resp = keyboard.Keyboard()

# Initialize components for Routine "isi_fixation_cross"
isi_fixation_crossClock = core.Clock()
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.05, 0.05),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
    opacity=None, depth=0.0, interpolate=True)

# Initialize components for Routine "practice_train_trial"
practice_train_trialClock = core.Clock()
train_image_2 = visual.ImageStim(
    win=win,
    name='train_image_2', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
sound_train_2 = sound.Sound('A', secs=1.0, stereo=True, hamming=True,
    name='sound_train_2')
sound_train_2.setVolume(1.0)
train_text_2 = visual.TextStim(win=win, name='train_text_2',
    text="Press 'R' to repeat the sound and 'Z' to go to next trial",
    font='Open Sans',
    pos=(0, -0.25), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
practice_train_resp = keyboard.Keyboard()

# Initialize components for Routine "practice_ir_test_info"
practice_ir_test_infoClock = core.Clock()
prac_ir_test_text = visual.TextStim(win=win, name='prac_ir_test_text',
    text="Practice Test block: Image retrieval\n\nYou will now hear an audio word and you must choose the correct image using the digit keys on the keyboard\n\nPress 'Space' to start",
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
prac_ir_resp = keyboard.Keyboard()

# Initialize components for Routine "isi_fixation_cross"
isi_fixation_crossClock = core.Clock()
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.05, 0.05),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
    opacity=None, depth=0.0, interpolate=True)

# Initialize components for Routine "prac_ir_test_trial"
prac_ir_test_trialClock = core.Clock()
ir_prac_sound = sound.Sound('A', secs=1.0, stereo=True, hamming=True,
    name='ir_prac_sound')
ir_prac_sound.setVolume(1.0)
ir_prac_i1 = visual.ImageStim(
    win=win,
    name='ir_prac_i1', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(-0.35, 0.1), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
ir_prac_i2 = visual.ImageStim(
    win=win,
    name='ir_prac_i2', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0.35, 0.1), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
ir_prac_resp = keyboard.Keyboard()
ir_prac_pos2 = visual.TextStim(win=win, name='ir_prac_pos2',
    text='2',
    font='Open Sans',
    pos=(0.35, -0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-5.0);
ir_prac_pos1 = visual.TextStim(win=win, name='ir_prac_pos1',
    text='1',
    font='Open Sans',
    pos=(-0.35, -0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-6.0);
ir_prac_replay_text = visual.TextStim(win=win, name='ir_prac_replay_text',
    text="Press 'R' to replay the sound",
    font='Open Sans',
    pos=(0, -0.3), height=0.04, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-7.0);

# Initialize components for Routine "prac_ir_feedback"
prac_ir_feedbackClock = core.Clock()
ir_prac_feedback = visual.TextStim(win=win, name='ir_prac_feedback',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "experiment_information"
experiment_informationClock = core.Clock()
main_exp_ins_text = visual.TextStim(win=win, name='main_exp_ins_text',
    text='Main Experiment Blocks\n\nThis is the main experiment. Here you will hear 10 image-audio pairs in each subblock.\n\nAfter these 10 trials in a subblock you will be tested on some of the pairs. Rest of the pairs you will be tested on after all blocks are over.\n\nPress SPACE to start the experiment',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
main_exp_ins_resp = keyboard.Keyboard()

# Initialize components for Routine "block_info"
block_infoClock = core.Clock()
block_info_text = visual.TextStim(win=win, name='block_info_text',
    text="To start next block press 'SPACE'",
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
block_info_resp = keyboard.Keyboard()

# Initialize components for Routine "isi_fixation_cross"
isi_fixation_crossClock = core.Clock()
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.05, 0.05),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
    opacity=None, depth=0.0, interpolate=True)

# Initialize components for Routine "train_trial"
train_trialClock = core.Clock()
train_image = visual.ImageStim(
    win=win,
    name='train_image', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
sound_train = sound.Sound('A', secs=1.0, stereo=True, hamming=True,
    name='sound_train')
sound_train.setVolume(1.0)
train_text = visual.TextStim(win=win, name='train_text',
    text="Press 'R' to repeat the sound and 'Z' to go to next trial",
    font='Open Sans',
    pos=(0, -0.35), height=0.04, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
train_resp = keyboard.Keyboard()

# Initialize components for Routine "ir_test_ins_curr"
ir_test_ins_currClock = core.Clock()
ir_ins_text = visual.TextStim(win=win, name='ir_ins_text',
    text="Test Block\n\nHere you will hear one of the words you learned, and have to choose the object image that was paired with it. You have to click the corresponding digit from keyboard.\n\nYou can press 'R' to repeat the sound\n\nPress 'SPACE' to start ",
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
ir_ins_resp = keyboard.Keyboard()

# Initialize components for Routine "isi_fixation_cross"
isi_fixation_crossClock = core.Clock()
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.05, 0.05),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
    opacity=None, depth=0.0, interpolate=True)

# Initialize components for Routine "ir_test_trial_curr"
ir_test_trial_currClock = core.Clock()
test_image_1 = visual.ImageStim(
    win=win,
    name='test_image_1', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(-0.375, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
test_image_2 = visual.ImageStim(
    win=win,
    name='test_image_2', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
test_image_3 = visual.ImageStim(
    win=win,
    name='test_image_3', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0.375, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
test_image_4 = visual.ImageStim(
    win=win,
    name='test_image_4', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0.75, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
test_image_5 = visual.ImageStim(
    win=win,
    name='test_image_5', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(-0.75, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-4.0)
test_image_6 = visual.ImageStim(
    win=win,
    name='test_image_6', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(-0.375, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-5.0)
test_image_7 = visual.ImageStim(
    win=win,
    name='test_image_7', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-6.0)
test_image_8 = visual.ImageStim(
    win=win,
    name='test_image_8', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0.375, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-7.0)
test_image_9 = visual.ImageStim(
    win=win,
    name='test_image_9', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0.75, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-8.0)
test_image_0 = visual.ImageStim(
    win=win,
    name='test_image_0', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(-0.75, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-9.0)
pos_1 = visual.TextStim(win=win, name='pos_1',
    text='1',
    font='Open Sans',
    pos=(-0.375, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-10.0);
pos_2 = visual.TextStim(win=win, name='pos_2',
    text='2',
    font='Open Sans',
    pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-11.0);
pos_3 = visual.TextStim(win=win, name='pos_3',
    text='3',
    font='Open Sans',
    pos=(0.375, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-12.0);
pos_4 = visual.TextStim(win=win, name='pos_4',
    text='4',
    font='Open Sans',
    pos=(0.75, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-13.0);
pos_5 = visual.TextStim(win=win, name='pos_5',
    text='5',
    font='Open Sans',
    pos=(-0.75, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-14.0);
pos_6 = visual.TextStim(win=win, name='pos_6',
    text='6',
    font='Open Sans',
    pos=(-0.375, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-15.0);
pos_7 = visual.TextStim(win=win, name='pos_7',
    text='7',
    font='Open Sans',
    pos=(0, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-16.0);
pos_8 = visual.TextStim(win=win, name='pos_8',
    text='8',
    font='Open Sans',
    pos=(0.375, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-17.0);
pos_9 = visual.TextStim(win=win, name='pos_9',
    text='9',
    font='Open Sans',
    pos=(0.75, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-18.0);
pos_0 = visual.TextStim(win=win, name='pos_0',
    text='0',
    font='Open Sans',
    pos=(-0.75, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-19.0);
test_sound_ir = sound.Sound('A', secs=1.0, stereo=True, hamming=True,
    name='test_sound_ir')
test_sound_ir.setVolume(1.0)
key_resp_test = keyboard.Keyboard()
replay_text = visual.TextStim(win=win, name='replay_text',
    text="Press 'R' to replay the sound",
    font='Open Sans',
    pos=(0, -0.05), height=0.03, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-24.0);

# Initialize components for Routine "test_all_sessions"
test_all_sessionsClock = core.Clock()
main_exp_ins_text_2 = visual.TextStim(win=win, name='main_exp_ins_text_2',
    text='Test Blocks\n\nNow you will be tested on all the stimulis seen till now from all the blocks.\n\nPress SPACE to start ',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
main_exp_ins_resp_2 = keyboard.Keyboard()

# Initialize components for Routine "ir_test_ins_all"
ir_test_ins_allClock = core.Clock()
ir_ins_text_2 = visual.TextStim(win=win, name='ir_ins_text_2',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "isi_fixation_cross"
isi_fixation_crossClock = core.Clock()
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.05, 0.05),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[-1.0000, -1.0000, -1.0000],
    opacity=None, depth=0.0, interpolate=True)

# Initialize components for Routine "ir_test_trial_all"
ir_test_trial_allClock = core.Clock()
test_image = visual.ImageStim(
    win=win,
    name='test_image', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(-0.375, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
test_image_10 = visual.ImageStim(
    win=win,
    name='test_image_10', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
test_image_11 = visual.ImageStim(
    win=win,
    name='test_image_11', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0.375, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
test_image_12 = visual.ImageStim(
    win=win,
    name='test_image_12', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0.75, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
test_image_13 = visual.ImageStim(
    win=win,
    name='test_image_13', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(-0.75, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-4.0)
test_image_14 = visual.ImageStim(
    win=win,
    name='test_image_14', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(-0.375, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-5.0)
test_image_15 = visual.ImageStim(
    win=win,
    name='test_image_15', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-6.0)
test_image_16 = visual.ImageStim(
    win=win,
    name='test_image_16', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0.375, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-7.0)
test_image_17 = visual.ImageStim(
    win=win,
    name='test_image_17', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0.75, -0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-8.0)
test_image_18 = visual.ImageStim(
    win=win,
    name='test_image_18', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(-0.75, 0.25), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-9.0)
pos = visual.TextStim(win=win, name='pos',
    text='1',
    font='Open Sans',
    pos=(-0.375, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-10.0);
pos_10 = visual.TextStim(win=win, name='pos_10',
    text='2',
    font='Open Sans',
    pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-11.0);
pos_11 = visual.TextStim(win=win, name='pos_11',
    text='3',
    font='Open Sans',
    pos=(0.375, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-12.0);
pos_12 = visual.TextStim(win=win, name='pos_12',
    text='4',
    font='Open Sans',
    pos=(0.75, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-13.0);
pos_13 = visual.TextStim(win=win, name='pos_13',
    text='5',
    font='Open Sans',
    pos=(-0.75, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-14.0);
pos_14 = visual.TextStim(win=win, name='pos_14',
    text='6',
    font='Open Sans',
    pos=(-0.375, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-15.0);
pos_15 = visual.TextStim(win=win, name='pos_15',
    text='7',
    font='Open Sans',
    pos=(0, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-16.0);
pos_16 = visual.TextStim(win=win, name='pos_16',
    text='8',
    font='Open Sans',
    pos=(0.375, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-17.0);
pos_17 = visual.TextStim(win=win, name='pos_17',
    text='9',
    font='Open Sans',
    pos=(0.75, -0.4), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-18.0);
pos_18 = visual.TextStim(win=win, name='pos_18',
    text='0',
    font='Open Sans',
    pos=(-0.75, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-19.0);
test_sound_ir_2 = sound.Sound('A', secs=1.0, stereo=True, hamming=True,
    name='test_sound_ir_2')
test_sound_ir_2.setVolume(1.0)
key_resp_test_all = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "welcome_screen"-------
continueRoutine = True
# update component parameters for each repeat
welcome_resp.keys = []
welcome_resp.rt = []
_welcome_resp_allKeys = []
# keep track of which components have finished
welcome_screenComponents = [welcome_text, welcome_resp]
for thisComponent in welcome_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
welcome_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "welcome_screen"-------
while continueRoutine:
    # get current time
    t = welcome_screenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=welcome_screenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *welcome_text* updates
    if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        welcome_text.frameNStart = frameN  # exact frame index
        welcome_text.tStart = t  # local t and not account for scr refresh
        welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
        welcome_text.setAutoDraw(True)
    
    # *welcome_resp* updates
    waitOnFlip = False
    if welcome_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        welcome_resp.frameNStart = frameN  # exact frame index
        welcome_resp.tStart = t  # local t and not account for scr refresh
        welcome_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome_resp, 'tStartRefresh')  # time at next scr refresh
        welcome_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(welcome_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(welcome_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if welcome_resp.status == STARTED and not waitOnFlip:
        theseKeys = welcome_resp.getKeys(keyList=['space'], waitRelease=False)
        _welcome_resp_allKeys.extend(theseKeys)
        if len(_welcome_resp_allKeys):
            welcome_resp.keys = _welcome_resp_allKeys[-1].name  # just the last key pressed
            welcome_resp.rt = _welcome_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in welcome_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "welcome_screen"-------
for thisComponent in welcome_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if welcome_resp.keys in ['', [], None]:  # No response was made
    welcome_resp.keys = None
thisExp.addData('welcome_resp.keys',welcome_resp.keys)
if welcome_resp.keys != None:  # we had a response
    thisExp.addData('welcome_resp.rt', welcome_resp.rt)
thisExp.addData('welcome_resp.started', welcome_resp.tStartRefresh)
thisExp.addData('welcome_resp.stopped', welcome_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "welcome_screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
audio_check_reps = data.TrialHandler(nReps=50.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='audio_check_reps')
thisExp.addLoop(audio_check_reps)  # add the loop to the experiment
thisAudio_check_rep = audio_check_reps.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisAudio_check_rep.rgb)
if thisAudio_check_rep != None:
    for paramName in thisAudio_check_rep:
        exec('{} = thisAudio_check_rep[paramName]'.format(paramName))

for thisAudio_check_rep in audio_check_reps:
    currentLoop = audio_check_reps
    # abbreviate parameter names if possible (e.g. rgb = thisAudio_check_rep.rgb)
    if thisAudio_check_rep != None:
        for paramName in thisAudio_check_rep:
            exec('{} = thisAudio_check_rep[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "AudioCheck"-------
    continueRoutine = True
    # update component parameters for each repeat
    AudioResp.keys = []
    AudioResp.rt = []
    _AudioResp_allKeys = []
    audio_check_sound.setSound('audios/kinch_0.wav', secs=1.0, hamming=True)
    audio_check_sound.setVolume(1.0, log=False)
    # keep track of which components have finished
    AudioCheckComponents = [audio_check_text, AudioResp, audio_check_sound]
    for thisComponent in AudioCheckComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    AudioCheckClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "AudioCheck"-------
    while continueRoutine:
        # get current time
        t = AudioCheckClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=AudioCheckClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *audio_check_text* updates
        if audio_check_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            audio_check_text.frameNStart = frameN  # exact frame index
            audio_check_text.tStart = t  # local t and not account for scr refresh
            audio_check_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(audio_check_text, 'tStartRefresh')  # time at next scr refresh
            audio_check_text.setAutoDraw(True)
        
        # *AudioResp* updates
        waitOnFlip = False
        if AudioResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            AudioResp.frameNStart = frameN  # exact frame index
            AudioResp.tStart = t  # local t and not account for scr refresh
            AudioResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(AudioResp, 'tStartRefresh')  # time at next scr refresh
            AudioResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(AudioResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(AudioResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if AudioResp.status == STARTED and not waitOnFlip:
            theseKeys = AudioResp.getKeys(keyList=['r','space'], waitRelease=False)
            _AudioResp_allKeys.extend(theseKeys)
            if len(_AudioResp_allKeys):
                AudioResp.keys = _AudioResp_allKeys[-1].name  # just the last key pressed
                AudioResp.rt = _AudioResp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        # start/stop audio_check_sound
        if audio_check_sound.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            audio_check_sound.frameNStart = frameN  # exact frame index
            audio_check_sound.tStart = t  # local t and not account for scr refresh
            audio_check_sound.tStartRefresh = tThisFlipGlobal  # on global time
            audio_check_sound.play(when=win)  # sync with win flip
        if audio_check_sound.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > audio_check_sound.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                audio_check_sound.tStop = t  # not accounting for scr refresh
                audio_check_sound.frameNStop = frameN  # exact frame index
                win.timeOnFlip(audio_check_sound, 'tStopRefresh')  # time at next scr refresh
                audio_check_sound.stop()
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in AudioCheckComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "AudioCheck"-------
    for thisComponent in AudioCheckComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    audio_check_reps.addData('audio_check_text.started', audio_check_text.tStartRefresh)
    audio_check_reps.addData('audio_check_text.stopped', audio_check_text.tStopRefresh)
    # check responses
    if AudioResp.keys in ['', [], None]:  # No response was made
        AudioResp.keys = None
    audio_check_reps.addData('AudioResp.keys',AudioResp.keys)
    if AudioResp.keys != None:  # we had a response
        audio_check_reps.addData('AudioResp.rt', AudioResp.rt)
    audio_check_reps.addData('AudioResp.started', AudioResp.tStartRefresh)
    audio_check_reps.addData('AudioResp.stopped', AudioResp.tStopRefresh)
    audio_check_sound.stop()  # ensure sound has stopped at end of routine
    audio_check_reps.addData('audio_check_sound.started', audio_check_sound.tStartRefresh)
    audio_check_reps.addData('audio_check_sound.stopped', audio_check_sound.tStopRefresh)
    if 'space' in AudioResp.keys:
        audio_check_reps.finished = True
    # the Routine "AudioCheck" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 50.0 repeats of 'audio_check_reps'


# ------Prepare to start Routine "practice_info"-------
continueRoutine = True
# update component parameters for each repeat
practice_ins_resp.keys = []
practice_ins_resp.rt = []
_practice_ins_resp_allKeys = []
# keep track of which components have finished
practice_infoComponents = [practice_ins_text, practice_ins_resp]
for thisComponent in practice_infoComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
practice_infoClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "practice_info"-------
while continueRoutine:
    # get current time
    t = practice_infoClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=practice_infoClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *practice_ins_text* updates
    if practice_ins_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        practice_ins_text.frameNStart = frameN  # exact frame index
        practice_ins_text.tStart = t  # local t and not account for scr refresh
        practice_ins_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(practice_ins_text, 'tStartRefresh')  # time at next scr refresh
        practice_ins_text.setAutoDraw(True)
    
    # *practice_ins_resp* updates
    waitOnFlip = False
    if practice_ins_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        practice_ins_resp.frameNStart = frameN  # exact frame index
        practice_ins_resp.tStart = t  # local t and not account for scr refresh
        practice_ins_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(practice_ins_resp, 'tStartRefresh')  # time at next scr refresh
        practice_ins_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(practice_ins_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(practice_ins_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if practice_ins_resp.status == STARTED and not waitOnFlip:
        theseKeys = practice_ins_resp.getKeys(keyList=['space'], waitRelease=False)
        _practice_ins_resp_allKeys.extend(theseKeys)
        if len(_practice_ins_resp_allKeys):
            practice_ins_resp.keys = _practice_ins_resp_allKeys[-1].name  # just the last key pressed
            practice_ins_resp.rt = _practice_ins_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in practice_infoComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "practice_info"-------
for thisComponent in practice_infoComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('practice_ins_text.started', practice_ins_text.tStartRefresh)
thisExp.addData('practice_ins_text.stopped', practice_ins_text.tStopRefresh)
# check responses
if practice_ins_resp.keys in ['', [], None]:  # No response was made
    practice_ins_resp.keys = None
thisExp.addData('practice_ins_resp.keys',practice_ins_resp.keys)
if practice_ins_resp.keys != None:  # we had a response
    thisExp.addData('practice_ins_resp.rt', practice_ins_resp.rt)
thisExp.addData('practice_ins_resp.started', practice_ins_resp.tStartRefresh)
thisExp.addData('practice_ins_resp.stopped', practice_ins_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "practice_info" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
practice_train_loop = data.TrialHandler(nReps=1.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('practice_train.xlsx'),
    seed=None, name='practice_train_loop')
thisExp.addLoop(practice_train_loop)  # add the loop to the experiment
thisPractice_train_loop = practice_train_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisPractice_train_loop.rgb)
if thisPractice_train_loop != None:
    for paramName in thisPractice_train_loop:
        exec('{} = thisPractice_train_loop[paramName]'.format(paramName))

for thisPractice_train_loop in practice_train_loop:
    currentLoop = practice_train_loop
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_train_loop.rgb)
    if thisPractice_train_loop != None:
        for paramName in thisPractice_train_loop:
            exec('{} = thisPractice_train_loop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "isi_fixation_cross"-------
    continueRoutine = True
    routineTimer.add(0.300000)
    # update component parameters for each repeat
    # keep track of which components have finished
    isi_fixation_crossComponents = [polygon]
    for thisComponent in isi_fixation_crossComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    isi_fixation_crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "isi_fixation_cross"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = isi_fixation_crossClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=isi_fixation_crossClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *polygon* updates
        if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon.frameNStart = frameN  # exact frame index
            polygon.tStart = t  # local t and not account for scr refresh
            polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
            polygon.setAutoDraw(True)
        if polygon.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > polygon.tStartRefresh + 0.3-frameTolerance:
                # keep track of stop time/frame for later
                polygon.tStop = t  # not accounting for scr refresh
                polygon.frameNStop = frameN  # exact frame index
                win.timeOnFlip(polygon, 'tStopRefresh')  # time at next scr refresh
                polygon.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in isi_fixation_crossComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "isi_fixation_cross"-------
    for thisComponent in isi_fixation_crossComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_train_loop.addData('polygon.started', polygon.tStartRefresh)
    practice_train_loop.addData('polygon.stopped', polygon.tStopRefresh)
    
    # set up handler to look after randomisation of conditions etc
    practice_train_reps = data.TrialHandler(nReps=50.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='practice_train_reps')
    thisExp.addLoop(practice_train_reps)  # add the loop to the experiment
    thisPractice_train_rep = practice_train_reps.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_train_rep.rgb)
    if thisPractice_train_rep != None:
        for paramName in thisPractice_train_rep:
            exec('{} = thisPractice_train_rep[paramName]'.format(paramName))
    
    for thisPractice_train_rep in practice_train_reps:
        currentLoop = practice_train_reps
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_train_rep.rgb)
        if thisPractice_train_rep != None:
            for paramName in thisPractice_train_rep:
                exec('{} = thisPractice_train_rep[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "practice_train_trial"-------
        continueRoutine = True
        # update component parameters for each repeat
        train_image_2.setImage(image_train)
        sound_train_2.setSound(audio_train, secs=1.0, hamming=True)
        sound_train_2.setVolume(1.0, log=False)
        practice_train_resp.keys = []
        practice_train_resp.rt = []
        _practice_train_resp_allKeys = []
        # keep track of which components have finished
        practice_train_trialComponents = [train_image_2, sound_train_2, train_text_2, practice_train_resp]
        for thisComponent in practice_train_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        practice_train_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "practice_train_trial"-------
        while continueRoutine:
            # get current time
            t = practice_train_trialClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=practice_train_trialClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *train_image_2* updates
            if train_image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                train_image_2.frameNStart = frameN  # exact frame index
                train_image_2.tStart = t  # local t and not account for scr refresh
                train_image_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(train_image_2, 'tStartRefresh')  # time at next scr refresh
                train_image_2.setAutoDraw(True)
            # start/stop sound_train_2
            if sound_train_2.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                sound_train_2.frameNStart = frameN  # exact frame index
                sound_train_2.tStart = t  # local t and not account for scr refresh
                sound_train_2.tStartRefresh = tThisFlipGlobal  # on global time
                sound_train_2.play(when=win)  # sync with win flip
            if sound_train_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_train_2.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_train_2.tStop = t  # not accounting for scr refresh
                    sound_train_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_train_2, 'tStopRefresh')  # time at next scr refresh
                    sound_train_2.stop()
            
            # *train_text_2* updates
            if train_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                train_text_2.frameNStart = frameN  # exact frame index
                train_text_2.tStart = t  # local t and not account for scr refresh
                train_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(train_text_2, 'tStartRefresh')  # time at next scr refresh
                train_text_2.setAutoDraw(True)
            
            # *practice_train_resp* updates
            waitOnFlip = False
            if practice_train_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_train_resp.frameNStart = frameN  # exact frame index
                practice_train_resp.tStart = t  # local t and not account for scr refresh
                practice_train_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_train_resp, 'tStartRefresh')  # time at next scr refresh
                practice_train_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(practice_train_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(practice_train_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if practice_train_resp.status == STARTED and not waitOnFlip:
                theseKeys = practice_train_resp.getKeys(keyList=['z','r'], waitRelease=False)
                _practice_train_resp_allKeys.extend(theseKeys)
                if len(_practice_train_resp_allKeys):
                    practice_train_resp.keys = _practice_train_resp_allKeys[-1].name  # just the last key pressed
                    practice_train_resp.rt = _practice_train_resp_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_train_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "practice_train_trial"-------
        for thisComponent in practice_train_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        sound_train_2.stop()  # ensure sound has stopped at end of routine
        practice_train_reps.addData('sound_train_2.started', sound_train_2.tStartRefresh)
        practice_train_reps.addData('sound_train_2.stopped', sound_train_2.tStopRefresh)
        if 'z' in practice_train_resp.keys:
            practice_train_reps.finished = True
        # check responses
        if practice_train_resp.keys in ['', [], None]:  # No response was made
            practice_train_resp.keys = None
        practice_train_reps.addData('practice_train_resp.keys',practice_train_resp.keys)
        if practice_train_resp.keys != None:  # we had a response
            practice_train_reps.addData('practice_train_resp.rt', practice_train_resp.rt)
        practice_train_reps.addData('practice_train_resp.started', practice_train_resp.tStartRefresh)
        practice_train_reps.addData('practice_train_resp.stopped', practice_train_resp.tStopRefresh)
        # the Routine "practice_train_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 50.0 repeats of 'practice_train_reps'
    
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'practice_train_loop'


# ------Prepare to start Routine "practice_ir_test_info"-------
continueRoutine = True
# update component parameters for each repeat
prac_ir_resp.keys = []
prac_ir_resp.rt = []
_prac_ir_resp_allKeys = []
# keep track of which components have finished
practice_ir_test_infoComponents = [prac_ir_test_text, prac_ir_resp]
for thisComponent in practice_ir_test_infoComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
practice_ir_test_infoClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "practice_ir_test_info"-------
while continueRoutine:
    # get current time
    t = practice_ir_test_infoClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=practice_ir_test_infoClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *prac_ir_test_text* updates
    if prac_ir_test_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        prac_ir_test_text.frameNStart = frameN  # exact frame index
        prac_ir_test_text.tStart = t  # local t and not account for scr refresh
        prac_ir_test_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(prac_ir_test_text, 'tStartRefresh')  # time at next scr refresh
        prac_ir_test_text.setAutoDraw(True)
    
    # *prac_ir_resp* updates
    waitOnFlip = False
    if prac_ir_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        prac_ir_resp.frameNStart = frameN  # exact frame index
        prac_ir_resp.tStart = t  # local t and not account for scr refresh
        prac_ir_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(prac_ir_resp, 'tStartRefresh')  # time at next scr refresh
        prac_ir_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(prac_ir_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(prac_ir_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if prac_ir_resp.status == STARTED and not waitOnFlip:
        theseKeys = prac_ir_resp.getKeys(keyList=['space'], waitRelease=False)
        _prac_ir_resp_allKeys.extend(theseKeys)
        if len(_prac_ir_resp_allKeys):
            prac_ir_resp.keys = _prac_ir_resp_allKeys[-1].name  # just the last key pressed
            prac_ir_resp.rt = _prac_ir_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in practice_ir_test_infoComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "practice_ir_test_info"-------
for thisComponent in practice_ir_test_infoComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if prac_ir_resp.keys in ['', [], None]:  # No response was made
    prac_ir_resp.keys = None
thisExp.addData('prac_ir_resp.keys',prac_ir_resp.keys)
if prac_ir_resp.keys != None:  # we had a response
    thisExp.addData('prac_ir_resp.rt', prac_ir_resp.rt)
thisExp.addData('prac_ir_resp.started', prac_ir_resp.tStartRefresh)
thisExp.addData('prac_ir_resp.stopped', prac_ir_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "practice_ir_test_info" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
practice_ir_block = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('practice_ir_test.xlsx'),
    seed=None, name='practice_ir_block')
thisExp.addLoop(practice_ir_block)  # add the loop to the experiment
thisPractice_ir_block = practice_ir_block.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisPractice_ir_block.rgb)
if thisPractice_ir_block != None:
    for paramName in thisPractice_ir_block:
        exec('{} = thisPractice_ir_block[paramName]'.format(paramName))

for thisPractice_ir_block in practice_ir_block:
    currentLoop = practice_ir_block
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_ir_block.rgb)
    if thisPractice_ir_block != None:
        for paramName in thisPractice_ir_block:
            exec('{} = thisPractice_ir_block[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "isi_fixation_cross"-------
    continueRoutine = True
    routineTimer.add(0.300000)
    # update component parameters for each repeat
    # keep track of which components have finished
    isi_fixation_crossComponents = [polygon]
    for thisComponent in isi_fixation_crossComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    isi_fixation_crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "isi_fixation_cross"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = isi_fixation_crossClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=isi_fixation_crossClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *polygon* updates
        if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon.frameNStart = frameN  # exact frame index
            polygon.tStart = t  # local t and not account for scr refresh
            polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
            polygon.setAutoDraw(True)
        if polygon.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > polygon.tStartRefresh + 0.3-frameTolerance:
                # keep track of stop time/frame for later
                polygon.tStop = t  # not accounting for scr refresh
                polygon.frameNStop = frameN  # exact frame index
                win.timeOnFlip(polygon, 'tStopRefresh')  # time at next scr refresh
                polygon.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in isi_fixation_crossComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "isi_fixation_cross"-------
    for thisComponent in isi_fixation_crossComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_ir_block.addData('polygon.started', polygon.tStartRefresh)
    practice_ir_block.addData('polygon.stopped', polygon.tStopRefresh)
    
    # set up handler to look after randomisation of conditions etc
    practice_ir_loop = data.TrialHandler(nReps=50.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='practice_ir_loop')
    thisExp.addLoop(practice_ir_loop)  # add the loop to the experiment
    thisPractice_ir_loop = practice_ir_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_ir_loop.rgb)
    if thisPractice_ir_loop != None:
        for paramName in thisPractice_ir_loop:
            exec('{} = thisPractice_ir_loop[paramName]'.format(paramName))
    
    for thisPractice_ir_loop in practice_ir_loop:
        currentLoop = practice_ir_loop
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_ir_loop.rgb)
        if thisPractice_ir_loop != None:
            for paramName in thisPractice_ir_loop:
                exec('{} = thisPractice_ir_loop[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "prac_ir_test_trial"-------
        continueRoutine = True
        # update component parameters for each repeat
        ir_prac_sound.setSound(audio_test, secs=1.0, hamming=True)
        ir_prac_sound.setVolume(1.0, log=False)
        ir_prac_i1.setImage(image_test1)
        ir_prac_i2.setImage(image_test2)
        ir_prac_resp.keys = []
        ir_prac_resp.rt = []
        _ir_prac_resp_allKeys = []
        # keep track of which components have finished
        prac_ir_test_trialComponents = [ir_prac_sound, ir_prac_i1, ir_prac_i2, ir_prac_resp, ir_prac_pos2, ir_prac_pos1, ir_prac_replay_text]
        for thisComponent in prac_ir_test_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        prac_ir_test_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "prac_ir_test_trial"-------
        while continueRoutine:
            # get current time
            t = prac_ir_test_trialClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=prac_ir_test_trialClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # start/stop ir_prac_sound
            if ir_prac_sound.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                ir_prac_sound.frameNStart = frameN  # exact frame index
                ir_prac_sound.tStart = t  # local t and not account for scr refresh
                ir_prac_sound.tStartRefresh = tThisFlipGlobal  # on global time
                ir_prac_sound.play(when=win)  # sync with win flip
            if ir_prac_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ir_prac_sound.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    ir_prac_sound.tStop = t  # not accounting for scr refresh
                    ir_prac_sound.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(ir_prac_sound, 'tStopRefresh')  # time at next scr refresh
                    ir_prac_sound.stop()
            
            # *ir_prac_i1* updates
            if ir_prac_i1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ir_prac_i1.frameNStart = frameN  # exact frame index
                ir_prac_i1.tStart = t  # local t and not account for scr refresh
                ir_prac_i1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ir_prac_i1, 'tStartRefresh')  # time at next scr refresh
                ir_prac_i1.setAutoDraw(True)
            
            # *ir_prac_i2* updates
            if ir_prac_i2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ir_prac_i2.frameNStart = frameN  # exact frame index
                ir_prac_i2.tStart = t  # local t and not account for scr refresh
                ir_prac_i2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ir_prac_i2, 'tStartRefresh')  # time at next scr refresh
                ir_prac_i2.setAutoDraw(True)
            
            # *ir_prac_resp* updates
            waitOnFlip = False
            if ir_prac_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ir_prac_resp.frameNStart = frameN  # exact frame index
                ir_prac_resp.tStart = t  # local t and not account for scr refresh
                ir_prac_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ir_prac_resp, 'tStartRefresh')  # time at next scr refresh
                ir_prac_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(ir_prac_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(ir_prac_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if ir_prac_resp.status == STARTED and not waitOnFlip:
                theseKeys = ir_prac_resp.getKeys(keyList=['1','2','r','num_1','num_2'], waitRelease=False)
                _ir_prac_resp_allKeys.extend(theseKeys)
                if len(_ir_prac_resp_allKeys):
                    ir_prac_resp.keys = _ir_prac_resp_allKeys[-1].name  # just the last key pressed
                    ir_prac_resp.rt = _ir_prac_resp_allKeys[-1].rt
                    # was this correct?
                    if (ir_prac_resp.keys == str(corr_Ans)) or (ir_prac_resp.keys == corr_Ans):
                        ir_prac_resp.corr = 1
                    else:
                        ir_prac_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *ir_prac_pos2* updates
            if ir_prac_pos2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ir_prac_pos2.frameNStart = frameN  # exact frame index
                ir_prac_pos2.tStart = t  # local t and not account for scr refresh
                ir_prac_pos2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ir_prac_pos2, 'tStartRefresh')  # time at next scr refresh
                ir_prac_pos2.setAutoDraw(True)
            
            # *ir_prac_pos1* updates
            if ir_prac_pos1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ir_prac_pos1.frameNStart = frameN  # exact frame index
                ir_prac_pos1.tStart = t  # local t and not account for scr refresh
                ir_prac_pos1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ir_prac_pos1, 'tStartRefresh')  # time at next scr refresh
                ir_prac_pos1.setAutoDraw(True)
            
            # *ir_prac_replay_text* updates
            if ir_prac_replay_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ir_prac_replay_text.frameNStart = frameN  # exact frame index
                ir_prac_replay_text.tStart = t  # local t and not account for scr refresh
                ir_prac_replay_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ir_prac_replay_text, 'tStartRefresh')  # time at next scr refresh
                ir_prac_replay_text.setAutoDraw(True)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_ir_test_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "prac_ir_test_trial"-------
        for thisComponent in prac_ir_test_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        ir_prac_sound.stop()  # ensure sound has stopped at end of routine
        practice_ir_loop.addData('ir_prac_sound.started', ir_prac_sound.tStartRefresh)
        practice_ir_loop.addData('ir_prac_sound.stopped', ir_prac_sound.tStopRefresh)
        # check responses
        if ir_prac_resp.keys in ['', [], None]:  # No response was made
            ir_prac_resp.keys = None
            # was no response the correct answer?!
            if str(corr_Ans).lower() == 'none':
               ir_prac_resp.corr = 1;  # correct non-response
            else:
               ir_prac_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for practice_ir_loop (TrialHandler)
        practice_ir_loop.addData('ir_prac_resp.keys',ir_prac_resp.keys)
        practice_ir_loop.addData('ir_prac_resp.corr', ir_prac_resp.corr)
        if ir_prac_resp.keys != None:  # we had a response
            practice_ir_loop.addData('ir_prac_resp.rt', ir_prac_resp.rt)
        practice_ir_loop.addData('ir_prac_resp.started', ir_prac_resp.tStartRefresh)
        practice_ir_loop.addData('ir_prac_resp.stopped', ir_prac_resp.tStopRefresh)
        if '1' in ir_prac_resp.keys:
            practice_ir_loop.finished = True
            
        if '2' in ir_prac_resp.keys:
            practice_ir_loop.finished = True
        
        number = ir_prac_resp.keys[-1]
        
        if number == str(corr_Ans):
            msg="Correct!"
        else:
            msg="Oops! That was wrong"
        # the Routine "prac_ir_test_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 50.0 repeats of 'practice_ir_loop'
    
    
    # ------Prepare to start Routine "prac_ir_feedback"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    ir_prac_feedback.setText(msg)
    # keep track of which components have finished
    prac_ir_feedbackComponents = [ir_prac_feedback]
    for thisComponent in prac_ir_feedbackComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    prac_ir_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "prac_ir_feedback"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = prac_ir_feedbackClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=prac_ir_feedbackClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ir_prac_feedback* updates
        if ir_prac_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ir_prac_feedback.frameNStart = frameN  # exact frame index
            ir_prac_feedback.tStart = t  # local t and not account for scr refresh
            ir_prac_feedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ir_prac_feedback, 'tStartRefresh')  # time at next scr refresh
            ir_prac_feedback.setAutoDraw(True)
        if ir_prac_feedback.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > ir_prac_feedback.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                ir_prac_feedback.tStop = t  # not accounting for scr refresh
                ir_prac_feedback.frameNStop = frameN  # exact frame index
                win.timeOnFlip(ir_prac_feedback, 'tStopRefresh')  # time at next scr refresh
                ir_prac_feedback.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_ir_feedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "prac_ir_feedback"-------
    for thisComponent in prac_ir_feedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    practice_ir_block.addData('ir_prac_feedback.started', ir_prac_feedback.tStartRefresh)
    practice_ir_block.addData('ir_prac_feedback.stopped', ir_prac_feedback.tStopRefresh)
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'practice_ir_block'


# ------Prepare to start Routine "experiment_information"-------
continueRoutine = True
# update component parameters for each repeat
main_exp_ins_resp.keys = []
main_exp_ins_resp.rt = []
_main_exp_ins_resp_allKeys = []
# keep track of which components have finished
experiment_informationComponents = [main_exp_ins_text, main_exp_ins_resp]
for thisComponent in experiment_informationComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
experiment_informationClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "experiment_information"-------
while continueRoutine:
    # get current time
    t = experiment_informationClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=experiment_informationClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *main_exp_ins_text* updates
    if main_exp_ins_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        main_exp_ins_text.frameNStart = frameN  # exact frame index
        main_exp_ins_text.tStart = t  # local t and not account for scr refresh
        main_exp_ins_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(main_exp_ins_text, 'tStartRefresh')  # time at next scr refresh
        main_exp_ins_text.setAutoDraw(True)
    
    # *main_exp_ins_resp* updates
    waitOnFlip = False
    if main_exp_ins_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        main_exp_ins_resp.frameNStart = frameN  # exact frame index
        main_exp_ins_resp.tStart = t  # local t and not account for scr refresh
        main_exp_ins_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(main_exp_ins_resp, 'tStartRefresh')  # time at next scr refresh
        main_exp_ins_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(main_exp_ins_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(main_exp_ins_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if main_exp_ins_resp.status == STARTED and not waitOnFlip:
        theseKeys = main_exp_ins_resp.getKeys(keyList=['space'], waitRelease=False)
        _main_exp_ins_resp_allKeys.extend(theseKeys)
        if len(_main_exp_ins_resp_allKeys):
            main_exp_ins_resp.keys = _main_exp_ins_resp_allKeys[-1].name  # just the last key pressed
            main_exp_ins_resp.rt = _main_exp_ins_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in experiment_informationComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "experiment_information"-------
for thisComponent in experiment_informationComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('main_exp_ins_text.started', main_exp_ins_text.tStartRefresh)
thisExp.addData('main_exp_ins_text.stopped', main_exp_ins_text.tStopRefresh)
# check responses
if main_exp_ins_resp.keys in ['', [], None]:  # No response was made
    main_exp_ins_resp.keys = None
thisExp.addData('main_exp_ins_resp.keys',main_exp_ins_resp.keys)
if main_exp_ins_resp.keys != None:  # we had a response
    thisExp.addData('main_exp_ins_resp.rt', main_exp_ins_resp.rt)
thisExp.addData('main_exp_ins_resp.started', main_exp_ins_resp.tStartRefresh)
thisExp.addData('main_exp_ins_resp.stopped', main_exp_ins_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "experiment_information" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
block_loop = data.TrialHandler(nReps=1.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('curr_test_conditions.xlsx'),
    seed=None, name='block_loop')
thisExp.addLoop(block_loop)  # add the loop to the experiment
thisBlock_loop = block_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
if thisBlock_loop != None:
    for paramName in thisBlock_loop:
        exec('{} = thisBlock_loop[paramName]'.format(paramName))

for thisBlock_loop in block_loop:
    currentLoop = block_loop
    # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
    if thisBlock_loop != None:
        for paramName in thisBlock_loop:
            exec('{} = thisBlock_loop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "block_info"-------
    continueRoutine = True
    # update component parameters for each repeat
    block_info_resp.keys = []
    block_info_resp.rt = []
    _block_info_resp_allKeys = []
    # keep track of which components have finished
    block_infoComponents = [block_info_text, block_info_resp]
    for thisComponent in block_infoComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    block_infoClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "block_info"-------
    while continueRoutine:
        # get current time
        t = block_infoClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=block_infoClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *block_info_text* updates
        if block_info_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            block_info_text.frameNStart = frameN  # exact frame index
            block_info_text.tStart = t  # local t and not account for scr refresh
            block_info_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(block_info_text, 'tStartRefresh')  # time at next scr refresh
            block_info_text.setAutoDraw(True)
        
        # *block_info_resp* updates
        waitOnFlip = False
        if block_info_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            block_info_resp.frameNStart = frameN  # exact frame index
            block_info_resp.tStart = t  # local t and not account for scr refresh
            block_info_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(block_info_resp, 'tStartRefresh')  # time at next scr refresh
            block_info_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(block_info_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(block_info_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if block_info_resp.status == STARTED and not waitOnFlip:
            theseKeys = block_info_resp.getKeys(keyList=['space'], waitRelease=False)
            _block_info_resp_allKeys.extend(theseKeys)
            if len(_block_info_resp_allKeys):
                block_info_resp.keys = _block_info_resp_allKeys[-1].name  # just the last key pressed
                block_info_resp.rt = _block_info_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in block_infoComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "block_info"-------
    for thisComponent in block_infoComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    block_loop.addData('block_info_text.started', block_info_text.tStartRefresh)
    block_loop.addData('block_info_text.stopped', block_info_text.tStopRefresh)
    # check responses
    if block_info_resp.keys in ['', [], None]:  # No response was made
        block_info_resp.keys = None
    block_loop.addData('block_info_resp.keys',block_info_resp.keys)
    if block_info_resp.keys != None:  # we had a response
        block_loop.addData('block_info_resp.rt', block_info_resp.rt)
    block_loop.addData('block_info_resp.started', block_info_resp.tStartRefresh)
    block_loop.addData('block_info_resp.stopped', block_info_resp.tStopRefresh)
    # the Routine "block_info" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    train_loop = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(train_conds_file),
        seed=None, name='train_loop')
    thisExp.addLoop(train_loop)  # add the loop to the experiment
    thisTrain_loop = train_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrain_loop.rgb)
    if thisTrain_loop != None:
        for paramName in thisTrain_loop:
            exec('{} = thisTrain_loop[paramName]'.format(paramName))
    
    for thisTrain_loop in train_loop:
        currentLoop = train_loop
        # abbreviate parameter names if possible (e.g. rgb = thisTrain_loop.rgb)
        if thisTrain_loop != None:
            for paramName in thisTrain_loop:
                exec('{} = thisTrain_loop[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "isi_fixation_cross"-------
        continueRoutine = True
        routineTimer.add(0.300000)
        # update component parameters for each repeat
        # keep track of which components have finished
        isi_fixation_crossComponents = [polygon]
        for thisComponent in isi_fixation_crossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        isi_fixation_crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "isi_fixation_cross"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = isi_fixation_crossClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=isi_fixation_crossClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                polygon.setAutoDraw(True)
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(polygon, 'tStopRefresh')  # time at next scr refresh
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in isi_fixation_crossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "isi_fixation_cross"-------
        for thisComponent in isi_fixation_crossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        train_loop.addData('polygon.started', polygon.tStartRefresh)
        train_loop.addData('polygon.stopped', polygon.tStopRefresh)
        
        # set up handler to look after randomisation of conditions etc
        train_repeat = data.TrialHandler(nReps=50.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='train_repeat')
        thisExp.addLoop(train_repeat)  # add the loop to the experiment
        thisTrain_repeat = train_repeat.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrain_repeat.rgb)
        if thisTrain_repeat != None:
            for paramName in thisTrain_repeat:
                exec('{} = thisTrain_repeat[paramName]'.format(paramName))
        
        for thisTrain_repeat in train_repeat:
            currentLoop = train_repeat
            # abbreviate parameter names if possible (e.g. rgb = thisTrain_repeat.rgb)
            if thisTrain_repeat != None:
                for paramName in thisTrain_repeat:
                    exec('{} = thisTrain_repeat[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "train_trial"-------
            continueRoutine = True
            # update component parameters for each repeat
            train_image.setImage(image_train)
            sound_train.setSound(audio_train, secs=1.0, hamming=True)
            sound_train.setVolume(1.0, log=False)
            # INITIALIZE variables
            exitFlag = 0
            train_resp.keys = []
            train_resp.rt = []
            _train_resp_allKeys = []
            # keep track of which components have finished
            train_trialComponents = [train_image, sound_train, train_text, train_resp]
            for thisComponent in train_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            train_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "train_trial"-------
            while continueRoutine:
                # get current time
                t = train_trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=train_trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *train_image* updates
                if train_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    train_image.frameNStart = frameN  # exact frame index
                    train_image.tStart = t  # local t and not account for scr refresh
                    train_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(train_image, 'tStartRefresh')  # time at next scr refresh
                    train_image.setAutoDraw(True)
                # start/stop sound_train
                if sound_train.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                    # keep track of start time/frame for later
                    sound_train.frameNStart = frameN  # exact frame index
                    sound_train.tStart = t  # local t and not account for scr refresh
                    sound_train.tStartRefresh = tThisFlipGlobal  # on global time
                    sound_train.play(when=win)  # sync with win flip
                if sound_train.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_train.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        sound_train.tStop = t  # not accounting for scr refresh
                        sound_train.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(sound_train, 'tStopRefresh')  # time at next scr refresh
                        sound_train.stop()
                
                # *train_text* updates
                if train_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    train_text.frameNStart = frameN  # exact frame index
                    train_text.tStart = t  # local t and not account for scr refresh
                    train_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(train_text, 'tStartRefresh')  # time at next scr refresh
                    train_text.setAutoDraw(True)
                if exitFlag == 1:
                    continueRoutine = False
                
                # *train_resp* updates
                waitOnFlip = False
                if train_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    train_resp.frameNStart = frameN  # exact frame index
                    train_resp.tStart = t  # local t and not account for scr refresh
                    train_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(train_resp, 'tStartRefresh')  # time at next scr refresh
                    train_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(train_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(train_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if train_resp.status == STARTED and not waitOnFlip:
                    theseKeys = train_resp.getKeys(keyList=['r','z','q','p'], waitRelease=False)
                    _train_resp_allKeys.extend(theseKeys)
                    if len(_train_resp_allKeys):
                        train_resp.keys = _train_resp_allKeys[-1].name  # just the last key pressed
                        train_resp.rt = _train_resp_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in train_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "train_trial"-------
            for thisComponent in train_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            train_repeat.addData('train_image.started', train_image.tStartRefresh)
            train_repeat.addData('train_image.stopped', train_image.tStopRefresh)
            sound_train.stop()  # ensure sound has stopped at end of routine
            train_repeat.addData('sound_train.started', sound_train.tStartRefresh)
            train_repeat.addData('sound_train.stopped', sound_train.tStopRefresh)
            if 'z' in train_resp.keys:
                train_repeat.finished = True
                
            # QUIT the block if Q pressed
            if 'q' in train_resp.keys:
                train_repeat.finished = True
                train_loop.finished = True
                continueRoutine     = False
                exitFlag            = 1
            # check responses
            if train_resp.keys in ['', [], None]:  # No response was made
                train_resp.keys = None
            train_repeat.addData('train_resp.keys',train_resp.keys)
            if train_resp.keys != None:  # we had a response
                train_repeat.addData('train_resp.rt', train_resp.rt)
            train_repeat.addData('train_resp.started', train_resp.tStartRefresh)
            train_repeat.addData('train_resp.stopped', train_resp.tStopRefresh)
            # the Routine "train_trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 50.0 repeats of 'train_repeat'
        
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'train_loop'
    
    
    # ------Prepare to start Routine "ir_test_ins_curr"-------
    continueRoutine = True
    # update component parameters for each repeat
    ir_ins_resp.keys = []
    ir_ins_resp.rt = []
    _ir_ins_resp_allKeys = []
    # INITIALIZE variables
    exitFlag = 0
    ir_block_corr = 0
    # keep track of which components have finished
    ir_test_ins_currComponents = [ir_ins_text, ir_ins_resp]
    for thisComponent in ir_test_ins_currComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    ir_test_ins_currClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "ir_test_ins_curr"-------
    while continueRoutine:
        # get current time
        t = ir_test_ins_currClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=ir_test_ins_currClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ir_ins_text* updates
        if ir_ins_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ir_ins_text.frameNStart = frameN  # exact frame index
            ir_ins_text.tStart = t  # local t and not account for scr refresh
            ir_ins_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ir_ins_text, 'tStartRefresh')  # time at next scr refresh
            ir_ins_text.setAutoDraw(True)
        
        # *ir_ins_resp* updates
        waitOnFlip = False
        if ir_ins_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ir_ins_resp.frameNStart = frameN  # exact frame index
            ir_ins_resp.tStart = t  # local t and not account for scr refresh
            ir_ins_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ir_ins_resp, 'tStartRefresh')  # time at next scr refresh
            ir_ins_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(ir_ins_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(ir_ins_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if ir_ins_resp.status == STARTED and not waitOnFlip:
            theseKeys = ir_ins_resp.getKeys(keyList=['space','p','q'], waitRelease=False)
            _ir_ins_resp_allKeys.extend(theseKeys)
            if len(_ir_ins_resp_allKeys):
                ir_ins_resp.keys = _ir_ins_resp_allKeys[-1].name  # just the last key pressed
                ir_ins_resp.rt = _ir_ins_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ir_test_ins_currComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "ir_test_ins_curr"-------
    for thisComponent in ir_test_ins_currComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    block_loop.addData('ir_ins_text.started', ir_ins_text.tStartRefresh)
    block_loop.addData('ir_ins_text.stopped', ir_ins_text.tStopRefresh)
    # check responses
    if ir_ins_resp.keys in ['', [], None]:  # No response was made
        ir_ins_resp.keys = None
    block_loop.addData('ir_ins_resp.keys',ir_ins_resp.keys)
    if ir_ins_resp.keys != None:  # we had a response
        block_loop.addData('ir_ins_resp.rt', ir_ins_resp.rt)
    block_loop.addData('ir_ins_resp.started', ir_ins_resp.tStartRefresh)
    block_loop.addData('ir_ins_resp.stopped', ir_ins_resp.tStopRefresh)
    # the Routine "ir_test_ins_curr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    ir_test_loop = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(ir_test_conds_file),
        seed=None, name='ir_test_loop')
    thisExp.addLoop(ir_test_loop)  # add the loop to the experiment
    thisIr_test_loop = ir_test_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIr_test_loop.rgb)
    if thisIr_test_loop != None:
        for paramName in thisIr_test_loop:
            exec('{} = thisIr_test_loop[paramName]'.format(paramName))
    
    for thisIr_test_loop in ir_test_loop:
        currentLoop = ir_test_loop
        # abbreviate parameter names if possible (e.g. rgb = thisIr_test_loop.rgb)
        if thisIr_test_loop != None:
            for paramName in thisIr_test_loop:
                exec('{} = thisIr_test_loop[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "isi_fixation_cross"-------
        continueRoutine = True
        routineTimer.add(0.300000)
        # update component parameters for each repeat
        # keep track of which components have finished
        isi_fixation_crossComponents = [polygon]
        for thisComponent in isi_fixation_crossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        isi_fixation_crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "isi_fixation_cross"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = isi_fixation_crossClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=isi_fixation_crossClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                polygon.setAutoDraw(True)
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(polygon, 'tStopRefresh')  # time at next scr refresh
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in isi_fixation_crossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "isi_fixation_cross"-------
        for thisComponent in isi_fixation_crossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        ir_test_loop.addData('polygon.started', polygon.tStartRefresh)
        ir_test_loop.addData('polygon.stopped', polygon.tStopRefresh)
        
        # set up handler to look after randomisation of conditions etc
        image_test_repeat = data.TrialHandler(nReps=50.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='image_test_repeat')
        thisExp.addLoop(image_test_repeat)  # add the loop to the experiment
        thisImage_test_repeat = image_test_repeat.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisImage_test_repeat.rgb)
        if thisImage_test_repeat != None:
            for paramName in thisImage_test_repeat:
                exec('{} = thisImage_test_repeat[paramName]'.format(paramName))
        
        for thisImage_test_repeat in image_test_repeat:
            currentLoop = image_test_repeat
            # abbreviate parameter names if possible (e.g. rgb = thisImage_test_repeat.rgb)
            if thisImage_test_repeat != None:
                for paramName in thisImage_test_repeat:
                    exec('{} = thisImage_test_repeat[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "ir_test_trial_curr"-------
            continueRoutine = True
            # update component parameters for each repeat
            test_image_1.setImage(image_test2)
            test_image_2.setImage(image_test3)
            test_image_3.setImage(image_test4)
            test_image_4.setImage(image_test5)
            test_image_5.setImage(image_test6)
            test_image_6.setImage(image_test7)
            test_image_7.setImage(image_test8)
            test_image_8.setImage(image_test9)
            test_image_9.setImage(image_test10)
            test_image_0.setImage(image_test1)
            test_sound_ir.setSound(audio_test, secs=1.0, hamming=True)
            test_sound_ir.setVolume(1.0, log=False)
            key_resp_test.keys = []
            key_resp_test.rt = []
            _key_resp_test_allKeys = []
            # INITIALIZE variables
            exitFlag = 0
            isPaused = 0
            # keep track of which components have finished
            ir_test_trial_currComponents = [test_image_1, test_image_2, test_image_3, test_image_4, test_image_5, test_image_6, test_image_7, test_image_8, test_image_9, test_image_0, pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8, pos_9, pos_0, test_sound_ir, key_resp_test, replay_text]
            for thisComponent in ir_test_trial_currComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            ir_test_trial_currClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "ir_test_trial_curr"-------
            while continueRoutine:
                # get current time
                t = ir_test_trial_currClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=ir_test_trial_currClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *test_image_1* updates
                if test_image_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_1.frameNStart = frameN  # exact frame index
                    test_image_1.tStart = t  # local t and not account for scr refresh
                    test_image_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_1, 'tStartRefresh')  # time at next scr refresh
                    test_image_1.setAutoDraw(True)
                
                # *test_image_2* updates
                if test_image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_2.frameNStart = frameN  # exact frame index
                    test_image_2.tStart = t  # local t and not account for scr refresh
                    test_image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_2, 'tStartRefresh')  # time at next scr refresh
                    test_image_2.setAutoDraw(True)
                
                # *test_image_3* updates
                if test_image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_3.frameNStart = frameN  # exact frame index
                    test_image_3.tStart = t  # local t and not account for scr refresh
                    test_image_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_3, 'tStartRefresh')  # time at next scr refresh
                    test_image_3.setAutoDraw(True)
                
                # *test_image_4* updates
                if test_image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_4.frameNStart = frameN  # exact frame index
                    test_image_4.tStart = t  # local t and not account for scr refresh
                    test_image_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_4, 'tStartRefresh')  # time at next scr refresh
                    test_image_4.setAutoDraw(True)
                
                # *test_image_5* updates
                if test_image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_5.frameNStart = frameN  # exact frame index
                    test_image_5.tStart = t  # local t and not account for scr refresh
                    test_image_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_5, 'tStartRefresh')  # time at next scr refresh
                    test_image_5.setAutoDraw(True)
                
                # *test_image_6* updates
                if test_image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_6.frameNStart = frameN  # exact frame index
                    test_image_6.tStart = t  # local t and not account for scr refresh
                    test_image_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_6, 'tStartRefresh')  # time at next scr refresh
                    test_image_6.setAutoDraw(True)
                
                # *test_image_7* updates
                if test_image_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_7.frameNStart = frameN  # exact frame index
                    test_image_7.tStart = t  # local t and not account for scr refresh
                    test_image_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_7, 'tStartRefresh')  # time at next scr refresh
                    test_image_7.setAutoDraw(True)
                
                # *test_image_8* updates
                if test_image_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_8.frameNStart = frameN  # exact frame index
                    test_image_8.tStart = t  # local t and not account for scr refresh
                    test_image_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_8, 'tStartRefresh')  # time at next scr refresh
                    test_image_8.setAutoDraw(True)
                
                # *test_image_9* updates
                if test_image_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_9.frameNStart = frameN  # exact frame index
                    test_image_9.tStart = t  # local t and not account for scr refresh
                    test_image_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_9, 'tStartRefresh')  # time at next scr refresh
                    test_image_9.setAutoDraw(True)
                
                # *test_image_0* updates
                if test_image_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_0.frameNStart = frameN  # exact frame index
                    test_image_0.tStart = t  # local t and not account for scr refresh
                    test_image_0.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_0, 'tStartRefresh')  # time at next scr refresh
                    test_image_0.setAutoDraw(True)
                
                # *pos_1* updates
                if pos_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_1.frameNStart = frameN  # exact frame index
                    pos_1.tStart = t  # local t and not account for scr refresh
                    pos_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_1, 'tStartRefresh')  # time at next scr refresh
                    pos_1.setAutoDraw(True)
                
                # *pos_2* updates
                if pos_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_2.frameNStart = frameN  # exact frame index
                    pos_2.tStart = t  # local t and not account for scr refresh
                    pos_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_2, 'tStartRefresh')  # time at next scr refresh
                    pos_2.setAutoDraw(True)
                
                # *pos_3* updates
                if pos_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_3.frameNStart = frameN  # exact frame index
                    pos_3.tStart = t  # local t and not account for scr refresh
                    pos_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_3, 'tStartRefresh')  # time at next scr refresh
                    pos_3.setAutoDraw(True)
                
                # *pos_4* updates
                if pos_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_4.frameNStart = frameN  # exact frame index
                    pos_4.tStart = t  # local t and not account for scr refresh
                    pos_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_4, 'tStartRefresh')  # time at next scr refresh
                    pos_4.setAutoDraw(True)
                
                # *pos_5* updates
                if pos_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_5.frameNStart = frameN  # exact frame index
                    pos_5.tStart = t  # local t and not account for scr refresh
                    pos_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_5, 'tStartRefresh')  # time at next scr refresh
                    pos_5.setAutoDraw(True)
                
                # *pos_6* updates
                if pos_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_6.frameNStart = frameN  # exact frame index
                    pos_6.tStart = t  # local t and not account for scr refresh
                    pos_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_6, 'tStartRefresh')  # time at next scr refresh
                    pos_6.setAutoDraw(True)
                
                # *pos_7* updates
                if pos_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_7.frameNStart = frameN  # exact frame index
                    pos_7.tStart = t  # local t and not account for scr refresh
                    pos_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_7, 'tStartRefresh')  # time at next scr refresh
                    pos_7.setAutoDraw(True)
                
                # *pos_8* updates
                if pos_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_8.frameNStart = frameN  # exact frame index
                    pos_8.tStart = t  # local t and not account for scr refresh
                    pos_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_8, 'tStartRefresh')  # time at next scr refresh
                    pos_8.setAutoDraw(True)
                
                # *pos_9* updates
                if pos_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_9.frameNStart = frameN  # exact frame index
                    pos_9.tStart = t  # local t and not account for scr refresh
                    pos_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_9, 'tStartRefresh')  # time at next scr refresh
                    pos_9.setAutoDraw(True)
                
                # *pos_0* updates
                if pos_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_0.frameNStart = frameN  # exact frame index
                    pos_0.tStart = t  # local t and not account for scr refresh
                    pos_0.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_0, 'tStartRefresh')  # time at next scr refresh
                    pos_0.setAutoDraw(True)
                # start/stop test_sound_ir
                if test_sound_ir.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                    # keep track of start time/frame for later
                    test_sound_ir.frameNStart = frameN  # exact frame index
                    test_sound_ir.tStart = t  # local t and not account for scr refresh
                    test_sound_ir.tStartRefresh = tThisFlipGlobal  # on global time
                    test_sound_ir.play(when=win)  # sync with win flip
                if test_sound_ir.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > test_sound_ir.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        test_sound_ir.tStop = t  # not accounting for scr refresh
                        test_sound_ir.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(test_sound_ir, 'tStopRefresh')  # time at next scr refresh
                        test_sound_ir.stop()
                
                # *key_resp_test* updates
                waitOnFlip = False
                if key_resp_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_test.frameNStart = frameN  # exact frame index
                    key_resp_test.tStart = t  # local t and not account for scr refresh
                    key_resp_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_test, 'tStartRefresh')  # time at next scr refresh
                    key_resp_test.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_test.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_test.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_test.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_test.getKeys(keyList=['1','2','3','4','5','r','0','6','7','8','9','p','q','num_0','num_1','num_2','num_3','num_4','num_5','num_6','num_7','num_8','num_9'], waitRelease=False)
                    _key_resp_test_allKeys.extend(theseKeys)
                    if len(_key_resp_test_allKeys):
                        key_resp_test.keys = _key_resp_test_allKeys[-1].name  # just the last key pressed
                        key_resp_test.rt = _key_resp_test_allKeys[-1].rt
                        # was this correct?
                        if (key_resp_test.keys == str(corrAns_ir)) or (key_resp_test.keys == corrAns_ir):
                            key_resp_test.corr = 1
                        else:
                            key_resp_test.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                if exitFlag == 1:
                    continueRoutine = False
                
                # *replay_text* updates
                if replay_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    replay_text.frameNStart = frameN  # exact frame index
                    replay_text.tStart = t  # local t and not account for scr refresh
                    replay_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(replay_text, 'tStartRefresh')  # time at next scr refresh
                    replay_text.setAutoDraw(True)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ir_test_trial_currComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "ir_test_trial_curr"-------
            for thisComponent in ir_test_trial_currComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            test_sound_ir.stop()  # ensure sound has stopped at end of routine
            image_test_repeat.addData('test_sound_ir.started', test_sound_ir.tStartRefresh)
            image_test_repeat.addData('test_sound_ir.stopped', test_sound_ir.tStopRefresh)
            # check responses
            if key_resp_test.keys in ['', [], None]:  # No response was made
                key_resp_test.keys = None
                # was no response the correct answer?!
                if str(corrAns_ir).lower() == 'none':
                   key_resp_test.corr = 1;  # correct non-response
                else:
                   key_resp_test.corr = 0;  # failed to respond (incorrectly)
            # store data for image_test_repeat (TrialHandler)
            image_test_repeat.addData('key_resp_test.keys',key_resp_test.keys)
            image_test_repeat.addData('key_resp_test.corr', key_resp_test.corr)
            if key_resp_test.keys != None:  # we had a response
                image_test_repeat.addData('key_resp_test.rt', key_resp_test.rt)
            image_test_repeat.addData('key_resp_test.started', key_resp_test.tStartRefresh)
            image_test_repeat.addData('key_resp_test.stopped', key_resp_test.tStopRefresh)
            if '1' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            if '2' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            if '3' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            if '4' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            if '5' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            if '6' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            if '7' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            if '8' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            if '9' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            if '0' in key_resp_test.keys:
                image_test_repeat.finished = True
                
            number = key_resp_test.keys[-1]
            
            if number == str(corrAns_ir):
                correct_ir =1
            else:
                correct_ir =0
              
            ir_block_corr = ir_block_corr+correct_ir
            
            thisExp.addData('corr_ir_trial',correct_ir)
            thisExp.addData('corr_ir_block',ir_block_corr)
            # QUIT the block if Q pressed
            if 'q' in key_resp_test.keys:
                ir_test_loop.finished = True
                image_test_repeat.finished = True
                continueRoutine     = False
                exitFlag            = 1
            image_test_repeat.addData('replay_text.started', replay_text.tStartRefresh)
            image_test_repeat.addData('replay_text.stopped', replay_text.tStopRefresh)
            # the Routine "ir_test_trial_curr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 50.0 repeats of 'image_test_repeat'
        
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'ir_test_loop'
    
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'block_loop'


# ------Prepare to start Routine "test_all_sessions"-------
continueRoutine = True
# update component parameters for each repeat
main_exp_ins_resp_2.keys = []
main_exp_ins_resp_2.rt = []
_main_exp_ins_resp_2_allKeys = []
# keep track of which components have finished
test_all_sessionsComponents = [main_exp_ins_text_2, main_exp_ins_resp_2]
for thisComponent in test_all_sessionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
test_all_sessionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "test_all_sessions"-------
while continueRoutine:
    # get current time
    t = test_all_sessionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=test_all_sessionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *main_exp_ins_text_2* updates
    if main_exp_ins_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        main_exp_ins_text_2.frameNStart = frameN  # exact frame index
        main_exp_ins_text_2.tStart = t  # local t and not account for scr refresh
        main_exp_ins_text_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(main_exp_ins_text_2, 'tStartRefresh')  # time at next scr refresh
        main_exp_ins_text_2.setAutoDraw(True)
    
    # *main_exp_ins_resp_2* updates
    waitOnFlip = False
    if main_exp_ins_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        main_exp_ins_resp_2.frameNStart = frameN  # exact frame index
        main_exp_ins_resp_2.tStart = t  # local t and not account for scr refresh
        main_exp_ins_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(main_exp_ins_resp_2, 'tStartRefresh')  # time at next scr refresh
        main_exp_ins_resp_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(main_exp_ins_resp_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(main_exp_ins_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if main_exp_ins_resp_2.status == STARTED and not waitOnFlip:
        theseKeys = main_exp_ins_resp_2.getKeys(keyList=['space'], waitRelease=False)
        _main_exp_ins_resp_2_allKeys.extend(theseKeys)
        if len(_main_exp_ins_resp_2_allKeys):
            main_exp_ins_resp_2.keys = _main_exp_ins_resp_2_allKeys[-1].name  # just the last key pressed
            main_exp_ins_resp_2.rt = _main_exp_ins_resp_2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in test_all_sessionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "test_all_sessions"-------
for thisComponent in test_all_sessionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('main_exp_ins_text_2.started', main_exp_ins_text_2.tStartRefresh)
thisExp.addData('main_exp_ins_text_2.stopped', main_exp_ins_text_2.tStopRefresh)
# check responses
if main_exp_ins_resp_2.keys in ['', [], None]:  # No response was made
    main_exp_ins_resp_2.keys = None
thisExp.addData('main_exp_ins_resp_2.keys',main_exp_ins_resp_2.keys)
if main_exp_ins_resp_2.keys != None:  # we had a response
    thisExp.addData('main_exp_ins_resp_2.rt', main_exp_ins_resp_2.rt)
thisExp.addData('main_exp_ins_resp_2.started', main_exp_ins_resp_2.tStartRefresh)
thisExp.addData('main_exp_ins_resp_2.stopped', main_exp_ins_resp_2.tStopRefresh)
thisExp.nextEntry()
# the Routine "test_all_sessions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
all_test_block = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('last_test_conditions.xlsx'),
    seed=None, name='all_test_block')
thisExp.addLoop(all_test_block)  # add the loop to the experiment
thisAll_test_block = all_test_block.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisAll_test_block.rgb)
if thisAll_test_block != None:
    for paramName in thisAll_test_block:
        exec('{} = thisAll_test_block[paramName]'.format(paramName))

for thisAll_test_block in all_test_block:
    currentLoop = all_test_block
    # abbreviate parameter names if possible (e.g. rgb = thisAll_test_block.rgb)
    if thisAll_test_block != None:
        for paramName in thisAll_test_block:
            exec('{} = thisAll_test_block[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "ir_test_ins_all"-------
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    # INITIALIZE variables
    exitFlag = 0
    ir_block_corr = 0
    # keep track of which components have finished
    ir_test_ins_allComponents = [ir_ins_text_2]
    for thisComponent in ir_test_ins_allComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    ir_test_ins_allClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "ir_test_ins_all"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = ir_test_ins_allClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=ir_test_ins_allClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ir_ins_text_2* updates
        if ir_ins_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ir_ins_text_2.frameNStart = frameN  # exact frame index
            ir_ins_text_2.tStart = t  # local t and not account for scr refresh
            ir_ins_text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ir_ins_text_2, 'tStartRefresh')  # time at next scr refresh
            ir_ins_text_2.setAutoDraw(True)
        if ir_ins_text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > ir_ins_text_2.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                ir_ins_text_2.tStop = t  # not accounting for scr refresh
                ir_ins_text_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(ir_ins_text_2, 'tStopRefresh')  # time at next scr refresh
                ir_ins_text_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ir_test_ins_allComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "ir_test_ins_all"-------
    for thisComponent in ir_test_ins_allComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    all_test_block.addData('ir_ins_text_2.started', ir_ins_text_2.tStartRefresh)
    all_test_block.addData('ir_ins_text_2.stopped', ir_ins_text_2.tStopRefresh)
    
    # set up handler to look after randomisation of conditions etc
    ir_test_all_loop = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(ir_conds_file),
        seed=None, name='ir_test_all_loop')
    thisExp.addLoop(ir_test_all_loop)  # add the loop to the experiment
    thisIr_test_all_loop = ir_test_all_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIr_test_all_loop.rgb)
    if thisIr_test_all_loop != None:
        for paramName in thisIr_test_all_loop:
            exec('{} = thisIr_test_all_loop[paramName]'.format(paramName))
    
    for thisIr_test_all_loop in ir_test_all_loop:
        currentLoop = ir_test_all_loop
        # abbreviate parameter names if possible (e.g. rgb = thisIr_test_all_loop.rgb)
        if thisIr_test_all_loop != None:
            for paramName in thisIr_test_all_loop:
                exec('{} = thisIr_test_all_loop[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "isi_fixation_cross"-------
        continueRoutine = True
        routineTimer.add(0.300000)
        # update component parameters for each repeat
        # keep track of which components have finished
        isi_fixation_crossComponents = [polygon]
        for thisComponent in isi_fixation_crossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        isi_fixation_crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "isi_fixation_cross"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = isi_fixation_crossClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=isi_fixation_crossClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                polygon.setAutoDraw(True)
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(polygon, 'tStopRefresh')  # time at next scr refresh
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in isi_fixation_crossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "isi_fixation_cross"-------
        for thisComponent in isi_fixation_crossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        ir_test_all_loop.addData('polygon.started', polygon.tStartRefresh)
        ir_test_all_loop.addData('polygon.stopped', polygon.tStopRefresh)
        
        # set up handler to look after randomisation of conditions etc
        image_test_rep2 = data.TrialHandler(nReps=50.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='image_test_rep2')
        thisExp.addLoop(image_test_rep2)  # add the loop to the experiment
        thisImage_test_rep2 = image_test_rep2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisImage_test_rep2.rgb)
        if thisImage_test_rep2 != None:
            for paramName in thisImage_test_rep2:
                exec('{} = thisImage_test_rep2[paramName]'.format(paramName))
        
        for thisImage_test_rep2 in image_test_rep2:
            currentLoop = image_test_rep2
            # abbreviate parameter names if possible (e.g. rgb = thisImage_test_rep2.rgb)
            if thisImage_test_rep2 != None:
                for paramName in thisImage_test_rep2:
                    exec('{} = thisImage_test_rep2[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "ir_test_trial_all"-------
            continueRoutine = True
            # update component parameters for each repeat
            test_image.setImage(image_test2)
            test_image_10.setImage(image_test3)
            test_image_11.setImage(image_test4)
            test_image_12.setImage(image_test5)
            test_image_13.setImage(image_test6)
            test_image_14.setImage(image_test7)
            test_image_15.setImage(image_test8)
            test_image_16.setImage(image_test9)
            test_image_17.setImage(image_test10)
            test_image_18.setImage(image_test1)
            test_sound_ir_2.setSound(audio_test, secs=1.0, hamming=True)
            test_sound_ir_2.setVolume(1.0, log=False)
            key_resp_test_all.keys = []
            key_resp_test_all.rt = []
            _key_resp_test_all_allKeys = []
            # INITIALIZE variables
            exitFlag = 0
            isPaused = 0
            # keep track of which components have finished
            ir_test_trial_allComponents = [test_image, test_image_10, test_image_11, test_image_12, test_image_13, test_image_14, test_image_15, test_image_16, test_image_17, test_image_18, pos, pos_10, pos_11, pos_12, pos_13, pos_14, pos_15, pos_16, pos_17, pos_18, test_sound_ir_2, key_resp_test_all]
            for thisComponent in ir_test_trial_allComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            ir_test_trial_allClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "ir_test_trial_all"-------
            while continueRoutine:
                # get current time
                t = ir_test_trial_allClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=ir_test_trial_allClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *test_image* updates
                if test_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image.frameNStart = frameN  # exact frame index
                    test_image.tStart = t  # local t and not account for scr refresh
                    test_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image, 'tStartRefresh')  # time at next scr refresh
                    test_image.setAutoDraw(True)
                
                # *test_image_10* updates
                if test_image_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_10.frameNStart = frameN  # exact frame index
                    test_image_10.tStart = t  # local t and not account for scr refresh
                    test_image_10.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_10, 'tStartRefresh')  # time at next scr refresh
                    test_image_10.setAutoDraw(True)
                
                # *test_image_11* updates
                if test_image_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_11.frameNStart = frameN  # exact frame index
                    test_image_11.tStart = t  # local t and not account for scr refresh
                    test_image_11.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_11, 'tStartRefresh')  # time at next scr refresh
                    test_image_11.setAutoDraw(True)
                
                # *test_image_12* updates
                if test_image_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_12.frameNStart = frameN  # exact frame index
                    test_image_12.tStart = t  # local t and not account for scr refresh
                    test_image_12.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_12, 'tStartRefresh')  # time at next scr refresh
                    test_image_12.setAutoDraw(True)
                
                # *test_image_13* updates
                if test_image_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_13.frameNStart = frameN  # exact frame index
                    test_image_13.tStart = t  # local t and not account for scr refresh
                    test_image_13.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_13, 'tStartRefresh')  # time at next scr refresh
                    test_image_13.setAutoDraw(True)
                
                # *test_image_14* updates
                if test_image_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_14.frameNStart = frameN  # exact frame index
                    test_image_14.tStart = t  # local t and not account for scr refresh
                    test_image_14.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_14, 'tStartRefresh')  # time at next scr refresh
                    test_image_14.setAutoDraw(True)
                
                # *test_image_15* updates
                if test_image_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_15.frameNStart = frameN  # exact frame index
                    test_image_15.tStart = t  # local t and not account for scr refresh
                    test_image_15.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_15, 'tStartRefresh')  # time at next scr refresh
                    test_image_15.setAutoDraw(True)
                
                # *test_image_16* updates
                if test_image_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_16.frameNStart = frameN  # exact frame index
                    test_image_16.tStart = t  # local t and not account for scr refresh
                    test_image_16.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_16, 'tStartRefresh')  # time at next scr refresh
                    test_image_16.setAutoDraw(True)
                
                # *test_image_17* updates
                if test_image_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_17.frameNStart = frameN  # exact frame index
                    test_image_17.tStart = t  # local t and not account for scr refresh
                    test_image_17.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_17, 'tStartRefresh')  # time at next scr refresh
                    test_image_17.setAutoDraw(True)
                
                # *test_image_18* updates
                if test_image_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    test_image_18.frameNStart = frameN  # exact frame index
                    test_image_18.tStart = t  # local t and not account for scr refresh
                    test_image_18.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(test_image_18, 'tStartRefresh')  # time at next scr refresh
                    test_image_18.setAutoDraw(True)
                
                # *pos* updates
                if pos.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos.frameNStart = frameN  # exact frame index
                    pos.tStart = t  # local t and not account for scr refresh
                    pos.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos, 'tStartRefresh')  # time at next scr refresh
                    pos.setAutoDraw(True)
                
                # *pos_10* updates
                if pos_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_10.frameNStart = frameN  # exact frame index
                    pos_10.tStart = t  # local t and not account for scr refresh
                    pos_10.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_10, 'tStartRefresh')  # time at next scr refresh
                    pos_10.setAutoDraw(True)
                
                # *pos_11* updates
                if pos_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_11.frameNStart = frameN  # exact frame index
                    pos_11.tStart = t  # local t and not account for scr refresh
                    pos_11.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_11, 'tStartRefresh')  # time at next scr refresh
                    pos_11.setAutoDraw(True)
                
                # *pos_12* updates
                if pos_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_12.frameNStart = frameN  # exact frame index
                    pos_12.tStart = t  # local t and not account for scr refresh
                    pos_12.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_12, 'tStartRefresh')  # time at next scr refresh
                    pos_12.setAutoDraw(True)
                
                # *pos_13* updates
                if pos_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_13.frameNStart = frameN  # exact frame index
                    pos_13.tStart = t  # local t and not account for scr refresh
                    pos_13.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_13, 'tStartRefresh')  # time at next scr refresh
                    pos_13.setAutoDraw(True)
                
                # *pos_14* updates
                if pos_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_14.frameNStart = frameN  # exact frame index
                    pos_14.tStart = t  # local t and not account for scr refresh
                    pos_14.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_14, 'tStartRefresh')  # time at next scr refresh
                    pos_14.setAutoDraw(True)
                
                # *pos_15* updates
                if pos_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_15.frameNStart = frameN  # exact frame index
                    pos_15.tStart = t  # local t and not account for scr refresh
                    pos_15.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_15, 'tStartRefresh')  # time at next scr refresh
                    pos_15.setAutoDraw(True)
                
                # *pos_16* updates
                if pos_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_16.frameNStart = frameN  # exact frame index
                    pos_16.tStart = t  # local t and not account for scr refresh
                    pos_16.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_16, 'tStartRefresh')  # time at next scr refresh
                    pos_16.setAutoDraw(True)
                
                # *pos_17* updates
                if pos_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_17.frameNStart = frameN  # exact frame index
                    pos_17.tStart = t  # local t and not account for scr refresh
                    pos_17.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_17, 'tStartRefresh')  # time at next scr refresh
                    pos_17.setAutoDraw(True)
                
                # *pos_18* updates
                if pos_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pos_18.frameNStart = frameN  # exact frame index
                    pos_18.tStart = t  # local t and not account for scr refresh
                    pos_18.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pos_18, 'tStartRefresh')  # time at next scr refresh
                    pos_18.setAutoDraw(True)
                # start/stop test_sound_ir_2
                if test_sound_ir_2.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                    # keep track of start time/frame for later
                    test_sound_ir_2.frameNStart = frameN  # exact frame index
                    test_sound_ir_2.tStart = t  # local t and not account for scr refresh
                    test_sound_ir_2.tStartRefresh = tThisFlipGlobal  # on global time
                    test_sound_ir_2.play(when=win)  # sync with win flip
                if test_sound_ir_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > test_sound_ir_2.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        test_sound_ir_2.tStop = t  # not accounting for scr refresh
                        test_sound_ir_2.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(test_sound_ir_2, 'tStopRefresh')  # time at next scr refresh
                        test_sound_ir_2.stop()
                
                # *key_resp_test_all* updates
                waitOnFlip = False
                if key_resp_test_all.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_test_all.frameNStart = frameN  # exact frame index
                    key_resp_test_all.tStart = t  # local t and not account for scr refresh
                    key_resp_test_all.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_test_all, 'tStartRefresh')  # time at next scr refresh
                    key_resp_test_all.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_test_all.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_test_all.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_test_all.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_test_all.getKeys(keyList=['1','2','3','4','5','r','0','6','7','8','9','p','q','num_0','num_1','num_2','num_3','num_4','num_5','num_6','num_7','num_8','num_9'], waitRelease=False)
                    _key_resp_test_all_allKeys.extend(theseKeys)
                    if len(_key_resp_test_all_allKeys):
                        key_resp_test_all.keys = _key_resp_test_all_allKeys[-1].name  # just the last key pressed
                        key_resp_test_all.rt = _key_resp_test_all_allKeys[-1].rt
                        # was this correct?
                        if (key_resp_test_all.keys == str(corrAns_ir)) or (key_resp_test_all.keys == corrAns_ir):
                            key_resp_test_all.corr = 1
                        else:
                            key_resp_test_all.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                if exitFlag == 1:
                    continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ir_test_trial_allComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "ir_test_trial_all"-------
            for thisComponent in ir_test_trial_allComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            test_sound_ir_2.stop()  # ensure sound has stopped at end of routine
            image_test_rep2.addData('test_sound_ir_2.started', test_sound_ir_2.tStartRefresh)
            image_test_rep2.addData('test_sound_ir_2.stopped', test_sound_ir_2.tStopRefresh)
            # check responses
            if key_resp_test_all.keys in ['', [], None]:  # No response was made
                key_resp_test_all.keys = None
                # was no response the correct answer?!
                if str(corrAns_ir).lower() == 'none':
                   key_resp_test_all.corr = 1;  # correct non-response
                else:
                   key_resp_test_all.corr = 0;  # failed to respond (incorrectly)
            # store data for image_test_rep2 (TrialHandler)
            image_test_rep2.addData('key_resp_test_all.keys',key_resp_test_all.keys)
            image_test_rep2.addData('key_resp_test_all.corr', key_resp_test_all.corr)
            if key_resp_test_all.keys != None:  # we had a response
                image_test_rep2.addData('key_resp_test_all.rt', key_resp_test_all.rt)
            image_test_rep2.addData('key_resp_test_all.started', key_resp_test_all.tStartRefresh)
            image_test_rep2.addData('key_resp_test_all.stopped', key_resp_test_all.tStopRefresh)
            if '1' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            if '2' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            if '3' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            if '4' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            if '5' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            if '6' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            if '7' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            if '8' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            if '9' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            if '0' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                
            number = key_resp_test_all.keys[-1]
            
            if number == str(corrAns_ir):
                correct_ir =1
            else:
                correct_ir =0
              
            ir_block_corr = ir_block_corr+correct_ir
            
            thisExp.addData('corr_ir_trial',correct_ir)
            thisExp.addData('corr_ir_block',ir_block_corr)
            # QUIT the block if Q pressed
            if 'q' in key_resp_test_all.keys:
                image_test_rep2.finished = True
                ir_test_all_loop.finished = True
                continueRoutine     = False
                exitFlag            = 1
            # the Routine "ir_test_trial_all" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 50.0 repeats of 'image_test_rep2'
        
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'ir_test_all_loop'
    
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'all_test_block'


# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
