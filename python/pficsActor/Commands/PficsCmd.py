#!/usr/bin/env python

import json
import base64
import numpy
import time

import opscore.protocols.keys as keys
import opscore.protocols.types as types

from opscore.utility.qstr import qstr

class PficsCmd(object):

    def __init__(self, actor):
        # This lets us access the rest of the actor.
        self.actor = actor

        # Declare the commands we implement. When the actor is started
        # these are registered with the parser, which will call the
        # associated methods when matched. The callbacks will be
        # passed a single argument, the parsed and typed command.
        #
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('setupField', 'fieldID', self.setupField),
            ('testloop', '<cnt> [<expTime>]', self.testloop),
            ('home', '<cnt> [<expTime>]', self.home),
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("pfics_pfics", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("fieldID", types.String(), 
                                                 help="ID for the field, which defines the fiber positions"),
                                        keys.Key("expTime", types.Float(), 
                                                 help="Seconds for exposure"))

    def ping(self, cmd):
        """Query the actor for liveness/happiness."""

        cmd.finish("text='Present and (probably) well'")

    def status(self, cmd):
        """Report status and version; obtain and send current data"""

        self.actor.sendVersionKey(cmd)

        keyStrings = ['text="nothing to say, really"']
        keyMsg = '; '.join(keyStrings)

        cmd.inform(keyMsg)
        cmd.diag('text="still nothing to say"')
        cmd.finish()


    def setupField(self, cmd):
        """ Fully configure all the fibers for the given field. """

        cmd.fail("text='Not yet implemented'")

    def home(self, cmd):
        """ Home the actuators. """

        cmd.fail("text='Not yet implemented'")

    def targetPositions(self, fieldName):
        """ return the (x,y) cobra positions for the given field.

        Obviously, you'd fetch from some database...
        """

        return numpy.random.random(9600).reshape(4800,2).astype('f4')
        
    def testloop(self, cmd):
        """ Run the expose-move loop a few times. For development. """

        cnt = cmd.cmd.keywords["cnt"].values[0]
        expTime = cmd.cmd.keywords["expTime"].values[0] \
          if "expTime" in cmd.cmd.keywords \
          else 0.0


        times = numpy.zeros((cnt, 4), dtype='f8')
        
        targetPos = self.targetPositions("some field ID")
        for i in range(cnt):
            times[i,0] = time.time()

            # Fetch measured centroid from the camera actor
            cmdString = "centroid expTime=%0.1f" % (expTime)
            cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                          forUserCmd=cmd, timeLim=expTime+5.0)
            if cmdVar.didFail:
                cmd.fail('text=%s' % (qstr('Failed to expose with %s' % (cmdString))))
                return
            # Encoding will be encapsulated.
            rawCentroids = self.actor.models['mcs'].keyVarDict['centroidsChunk'][0]
            centroids = numpy.fromstring(base64.b64decode(rawCentroids), dtype='f4').reshape(2400,2)
            times[i,1] = time.time()

            # Command the actuators to move.
            cmdString = 'moveTo chunk=%s' % (base64.b64encode(targetPos.tostring()))
            cmdVar = self.actor.cmdr.call(actor='mps', cmdStr=cmdString,
                                          forUserCmd=cmd, timeLim=5.0)
            if cmdVar.didFail:
                cmd.fail('text=%s' % (qstr('Failed to move with %s' % (cmdString))))
                return
            times[i,2] = time.time()

            cmdVar = self.actor.cmdr.call(actor='mps', cmdStr="ping",
                                          forUserCmd=cmd, timeLim=5.0)
            if cmdVar.didFail:
                cmd.fail('text=%s' % (qstr('Failed to ping')))
                return
            times[i,3] = time.time()

        for i, itimes in enumerate(times):
            cmd.inform('text="dt[%d]=%0.4f, %0.4f, %0.4f"' % (i+1, 
                                                              itimes[1]-itimes[0],
                                                              itimes[2]-itimes[1],
                                                              itimes[3]-itimes[2],
                                                              ))
        cmd.inform('text="dt[mean]=%0.4f, %0.4f, %0.4f"' % ((times[:,1]-times[:,0]).sum()/cnt,
                                                            (times[:,2]-times[:,1]).sum()/cnt,
                                                            (times[:,3]-times[:,2]).sum()/cnt))
                                                            
        cmd.finish()
