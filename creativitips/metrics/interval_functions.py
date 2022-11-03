#!/usr/bin/env python

notes = ['=C,', '^C,', '=D,', '^D,', '=E,', '=F,', '^F,', '=G,', '^G,', '=A,', '^A,', '=B,',
         '=C', '^C', '=D', '^D', '=E', '=F', '^F', '=G', '^G', '=A', '^A', '=B',
         '=c', '^c', '=d', '^d', '=e', '=f', '^f', '=g', '^g', '=a', '^a', '=b',
         '=c\'', '^c\'', '=d\'', '^d\'', '=e\'', '=f\'', '^f\'', '=g\'', '^g\'', '=a\'', '^a\'', '=b\'',
         '=c\'\'']

dictOfNotes = {notes[i]: i for i in range(0, len(notes))}

dict_of_accidentals = {"none": 0, "C": 0, "Am": 0, "GMix": 0, "DDor": 0, "EPhr": 0, "FLyd": 0, "BLoc": 0,
                       "G": 1, "Em": 1, "DMix": 1, "ADor": 1, "BPhr": 1, "CLyd": 1, "F#Loc": 1,
                       "D": 2, "Bm": 2, "AMix": 2, "EDor": 2, "F#Phr": 2, "GLyd": 2, "C#Loc": 2
                       }  # TODO: complete with the other keys/modes


# 7 sharps 	C# 	A#m 	G#Mix 	D#Dor 	E#Phr 	F#Lyd 	B#Loc
# 6 sharps 	F# 	D#m 	C#Mix 	G#Dor 	A#Phr 	BLyd 	E#Loc
# 5 sharps 	B 	G#m 	F#Mix 	C#Dor 	D#Phr 	ELyd 	A#Loc
# 4 sharps 	E 	C#m 	BMix 	F#Dor 	G#Phr 	ALyd 	D#Loc
# 3 sharps 	A 	F#m 	EMix 	BDor 	C#Phr 	DLyd 	G#Loc
# 2 sharps 	D 	Bm 	AMix 	EDor 	F#Phr 	GLyd 	C#Loc
# 1 sharp 	G 	Em 	DMix 	ADor 	BPhr 	CLyd 	F#Loc
# 0 sharps/flats 	C 	Am 	GMix 	DDor 	EPhr 	FLyd 	BLoc
# 1 flat 	F 	Dm 	CMix 	GDor 	APhr 	BbLyd 	ELoc
# 2 flats 	Bb 	Gm 	FMix 	CDor 	DPhr 	EbLyd 	ALoc
# 3 flats 	Eb 	Cm 	BbMix 	FDor 	GPhr 	AbLyd 	DLoc
# 4 flats 	Ab 	Fm 	EbMix 	BbDor 	CPhr 	DbLyd 	GLoc
# 5 flats 	Db 	Bbm 	AbMix 	EbDor 	FPhr 	GbLyd 	CLoc
# 6 flats 	Gb 	Ebm 	DbMix 	AbDor 	BbPhr 	CbLyd 	FLoc
# 7 flats 	Cb 	Abm 	GbMix 	DbDor 	EbPhr 	FbLyd 	BbLoc


############################################################################################################
def addsharps(melody, sharp_set):
    # Function adding sharps in given set of notes.
    for i in range(0, len(melody)):
        if melody[i][0] in sharp_set:
            melody[i] = '^' + melody[i]
        else:
            melody[i] = '=' + melody[i]


############################################################################################################
def key2accidentals(inputmelody, key):
    # Function returning the list of notes with accidentals corresponding to the given key.
    #
    # NOTE: input argument is a list of strings. key="none" means no key.
    # TODO: complete with other keys/modes

    nacc = dict_of_accidentals[key]

    melody = []
    newmelody = []

    # delete rests
    for i in range(0, len(inputmelody)):
        if not ('z' in inputmelody[i]):
            melody.append(inputmelody[i])

    if nacc == 0:
        for i in range(0, len(melody)):
            melody[i] = '=' + melody[i]

    if nacc == 1:
        addsharps(melody, "fF")

    if nacc == 2:
        addsharps(melody, "fFcC")

    return melody


############################################################################################################
def compute_intervals(melody):
    # Function returning the array of intervals in an abc melodic sequence.
    #
    # NOTE: the argument is a string containing a sequence of notes in abc notation WITH EXPLICIT ACCIDENTALS.

    codes = []

    for note in melody:
        mynote = (note.split('/'))[0]
        if mynote[len(mynote) - 1].isnumeric():
            mynote = mynote[0:len(mynote) - 1]
        codes.append(dictOfNotes[mynote])

    intervals = []
    for i in range(1, len(codes)):
        intervals.append(codes[i] - codes[i - 1])

    return intervals


