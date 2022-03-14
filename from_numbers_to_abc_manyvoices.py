#!/usr/bin/env python

# input: file with two lines, each with MIDI numbers for pitches

import sys


def clean_zeros(line):
    for i in range(2, len(line)):
        if int(line[i]) == 0 and int(line[i - 1]) != 0:
            line[i] = line[i - 1]


def convert_notes(line):
    voice = []
    i = 0
    while i < len(line) - 1:
        n1 = int(line[i])
        j = i + 1
        n2 = int(line[j])
        while n1 == n2 and j < len(line) - 1:
            j = j + 1
            n2 = int(line[j])
        d = j - i
        if d > 1:
            duration = str(d)
        else:
            duration = ''
        n = n1
        if n >= base:
            voice.append(notesAndAccidents[n - base] + duration)
        else:
            voice.append('z' + duration)
        i = j
    return voice


notesAndAccidents = ['=C,,', '^C,,', '=D,,', '^D,,', '=E,,', '=F,,', '^F,,', '=G,,', '^G,,', '=A,,', '^A,,', '=B,,',
                     '=C,', '^C,', '=D,', '^D,', '=E,', '=F,', '^F,', '=G,', '^G,', '=A,', '^A,', '=B,',
                     '=C', '^C', '=D', '^D', '=E', '=F', '^F', '=G', '^G', '=A', '^A', '=B',
                     '=c', '^c', '=d', '^d', '=e', '=f', '^f', '=g', '^g', '=a', '^a', '=b',
                     '=c\'', '^c\'', '=d\'', '^d\'', '=e\'', '=f\'', '^f\'', '=g\'', '^g\'', '=a\'', '^a\'', '=b\'',
                     '=c\'\'', '=d\'\'', '^d\'\'', '=e\'\'', '=f\'\'', '^f\'\'', '=g\'\'', '^g\'\'', '=a\'\'', '^a\'\'',
                     '=b\'\'',
                     '=c\'\'\'']

dictOfNotes = {notesAndAccidents[i]: i for i in range(0, len(notesAndAccidents))}

base = 36  # corresponds to C, in abc

with open(sys.argv[1], 'r') as f:
    data = [line.split() for line in f]
    nv = 0
    # for i in range(0,len(data)-1,2):
    for i in range(0, 2, 2):
        line = data[i][:]
        nv = nv + 1
        clean_zeros(line)
        voice = convert_notes(line)
        print('V:', str(nv), ' clef=treble staff=1', sep='')
        print('K:C')
        print('M:4/4')
        print('L:1/32')
        for n in voice:
            print(n, end=' ')
        print('\r')
