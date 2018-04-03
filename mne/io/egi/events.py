# -*- coding: utf-8 -*-
#
# License: BSD (3-clause)

from datetime import datetime
from glob import glob
from os.path import basename, join, splitext
from xml.etree.ElementTree import parse

import numpy as np

from ...utils import logger

def _get_epochs(filename):
    """Extract Epoch time data boundaries
    
    Parameters
    ----------
    filename : str
        File path
    """
    epochs = []
    #How do elegantly?
    xml_epochs = _parse_xml(join(filename,'epochs.xml'))
    for epoch in xml_epochs:
        #The division by 1000 converts from micro to seconds
        #This is because the timedelta objects used only have access to seconds
        tmpEp = {'beginTime':int(epoch['beginTime'])/1000,
                 'endTime': int(epoch['endTime'])/1000,
                 'firstBlock': int(epoch['firstBlock']),
                 'lastBlock': int(epoch['lastBlock']),
                }
        logger.info(tmpEp)
        if(not epochs):
            tmpEp['epochGap']=0
        else:
            tmpEp['epochGap']=tmpEp.get('beginTime')-epochs[-1].get('endTime')
        
        epochs.append(tmpEp)
    
    return epochs
    
def _get_epoch_corr(evtOnset,epochs):
    """Determines number of seconds to correct for based on breaks between epochs
    
    Parameters
    ----------
    evtOnset: long
        Onset of Event in Seconds
    epochs: dict
        Dictionary of Epochs
    """
    correction = 0
    logger.info("Start of correction loop.")
    for epoch in epochs:
        logger.info(epoch)
        logger.info("Event Onset Time: %s", evtOnset)
        logger.info("Epoch Begin Time: %s", epoch.get('beginTime'))
        logger.info("Epoch End Time: %s", epoch.get('endTime'))
        
        if(not(evtOnset > epoch.get('beginTime') and evtOnset < epoch.get('endTime'))):
            correction = correction + epoch.get('epochGap')
        else:
            logger.info("Return to calling method")
            return correction

def _read_events(input_fname, info, epochs):
    """Read events for the record.

    Parameters
    ----------
    input_fname : str
        The file path.
    info : dict
        Header info array.
    epochs : dict
        Epoch info dictionary
    """
    mff_events, event_codes = _read_mff_events(input_fname, info['sfreq'], epochs)
    info['n_events'] = len(event_codes)
    info['event_codes'] = np.asarray(event_codes).astype('<U4')
    events = np.zeros([info['n_events'],
                      info['n_segments'] * info['n_samples']])
    for n, event in enumerate(event_codes):
        for i in mff_events[event]:
            events[n][i] = n + 1
    return events, info


def _read_mff_events(filename, sfreq, epochs):
    """Extract the events.

    Parameters
    ----------
    filename : str
        File path.
    sfreq : float
        The sampling frequency
    epochs : dict
        Epoch info dictionary=
    """
    orig = {}
    
    for xml_file in glob(join(filename, '*.xml')):
        xml_type = splitext(basename(xml_file))[0]
        orig[xml_type] = _parse_xml(xml_file)
    xml_files = orig.keys()
    xml_events = [x for x in xml_files if x[:7] == 'Events_']
    for item in orig['info']:
        if 'recordTime' in item:
            start_time = _ns2py_time(item['recordTime'])
            break
    markers = []
    code = []
    for xml in xml_events:
        for event in orig[xml][2:]:
            event_start = _ns2py_time(event['beginTime'])
            #logger.info("Uncorrected Event time: %s", event_start)
            start = (event_start - start_time).total_seconds()
            #logger.info("Corrected Event time: %s",start * 1000)
            #Upconvert to Milliseconds from Seconds to interface  
            epoch_corr = _get_epoch_corr(start*1000,epochs)
            logger.info("Uncorrected Event Time is: %s", start)
            logger.info("Epoch Corrected Event Time is: %s",(start - epoch_corr))
            start = start - epoch_corr
            if event['code'] not in code:
                code.append(event['code'])
            marker = {'name': event['code'],
                      'start': start,
                      'start_sample': int(np.fix(start * sfreq)),
                      'end': start + float(event['duration']) / 1e9,
                      'chan': None,
                      }
            markers.append(marker)
    events_tims = dict()
    for ev in code:
        trig_samp = list(c['start_sample'] for n,
                         c in enumerate(markers) if c['name'] == ev)
        events_tims.update({ev: trig_samp})
    return events_tims, code


def _parse_xml(xml_file):
    """Parse XML file."""
    xml = parse(xml_file)
    root = xml.getroot()
    return _xml2list(root)


def _xml2list(root):
    """Parse XML item."""
    output = []
    for element in root:

        if len(element) > 0:
            if element[0].tag != element[-1].tag:
                output.append(_xml2dict(element))
            else:
                output.append(_xml2list(element))

        elif element.text:
            text = element.text.strip()
            if text:
                tag = _ns(element.tag)
                output.append({tag: text})

    return output


def _ns(s):
    """Remove namespace, but only if there is a namespace to begin with."""
    if '}' in s:
        return '}'.join(s.split('}')[1:])
    else:
        return s


def _xml2dict(root):
    """Use functions instead of Class.

    remove namespace based on
    http://stackoverflow.com/questions/2148119
    """
    output = {}
    if root.items():
        output.update(dict(root.items()))

    for element in root:
        if len(element) > 0:
            if len(element) == 1 or element[0].tag != element[1].tag:
                one_dict = _xml2dict(element)
            else:
                one_dict = {_ns(element[0].tag): _xml2list(element)}

            if element.items():
                one_dict.update(dict(element.items()))
            output.update({_ns(element.tag): one_dict})

        elif element.items():
            output.update({_ns(element.tag): dict(element.items())})

        else:
            output.update({_ns(element.tag): element.text})
    return output


def _ns2py_time(nstime):
    """Parse times."""
    nsdate = nstime[0:10]
    nstime0 = nstime[11:26]
    nstime00 = nsdate + " " + nstime0
    pytime = datetime.strptime(nstime00, '%Y-%m-%d %H:%M:%S.%f')
    return pytime


def _combine_triggers(data, remapping=None):
    """Combine binary triggers."""
    new_trigger = np.zeros(data.shape[1])
    if data.astype(bool).sum(axis=0).max() > 1:  # ensure no overlaps
        logger.info('    Found multiple events at the same time '
                    'sample. Cannot create trigger channel.')
        return
    if remapping is None:
        remapping = np.arange(data) + 1
    for d, event_id in zip(data, remapping):
        idx = d.nonzero()
        if np.any(idx):
            new_trigger[idx] += event_id
    return new_trigger
