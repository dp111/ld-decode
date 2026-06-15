/************************************************************************

    stacker.cpp

    ld-disc-stacker - Disc stacking for ld-decode
    Copyright (C) 2020-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-disc-stacker is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#include "stacker.h"
#include "stackingpool.h"

#include <cmath>

Stacker::Stacker(QAtomicInt& _abort, StackingPool& _stackingPool, QObject *parent)
    : QThread(parent), abort(_abort), stackingPool(_stackingPool)
{
}

void Stacker::run()
{
    // Variables for getInputFrame
    qint32 frameNumber;
    QVector<qint32> firstFieldSeqNo;
    QVector<qint32> secondFieldSeqNo;
    QVector<SourceVideo::Data> firstSourceField;
    QVector<SourceVideo::Data> secondSourceField;
    QVector<LdDecodeMetaData::Field> firstFieldMetadata;
    QVector<LdDecodeMetaData::Field> secondFieldMetadata;
    qint32 mode;
    qint32 smartThreshold;
    bool reverse;
    bool noDiffDod;
    bool passThrough;
    const bool& verbose = stackingPool.verbose;
    const bool& chromaAlign = stackingPool.chromaAlign;
    QVector<qint32> availableSourcesForFrame;

    while(!abort) {
        // Get the next field to process from the input file
        if (!stackingPool.getInputFrame(frameNumber, firstFieldSeqNo, firstSourceField, firstFieldMetadata,
                                       secondFieldSeqNo, secondSourceField, secondFieldMetadata,
                                       videoParameters, mode, smartThreshold, reverse, noDiffDod, passThrough,
                                       availableSourcesForFrame)) {
            // No more input fields -- exit
            break;
        }

        // Initialise the output fields and process sources to output
        SourceVideo::Data outputFirstField(firstSourceField[0].size());
        SourceVideo::Data outputSecondField(secondSourceField[0].size());
        DropOuts outputFirstFieldDropOuts;
        DropOuts outputSecondFieldDropOuts;

        stackField(frameNumber, firstSourceField, videoParameters[0], firstFieldMetadata, availableSourcesForFrame, noDiffDod, passThrough, outputFirstField, outputFirstFieldDropOuts, mode, smartThreshold, chromaAlign, verbose);
        stackField(frameNumber, secondSourceField, videoParameters[0], secondFieldMetadata, availableSourcesForFrame, noDiffDod, passThrough, outputSecondField, outputSecondFieldDropOuts, mode, smartThreshold, chromaAlign, verbose);

        // Return the processed fields
        stackingPool.setOutputFrame(frameNumber, outputFirstField, outputSecondField,
                                    firstFieldSeqNo[0], secondFieldSeqNo[0],
                                    outputFirstFieldDropOuts, outputSecondFieldDropOuts);
    }
}

// Method to stack fields
void Stacker::stackField(const qint32 frameNumber,const QVector<SourceVideo::Data>& inputFields,
                                      const LdDecodeMetaData::VideoParameters& videoParameters,
                                      const QVector<LdDecodeMetaData::Field>& fieldMetadata,
                                      const QVector<qint32> availableSourcesForFrame,
                                      const bool& noDiffDod,const bool& passThrough,
                                      SourceVideo::Data &outputField,
                                      DropOuts &dropOuts,
                                      const qint32& mode,
                                      const qint32& smartThreshold,
                                      const bool& chromaAlign,
                                      const bool& verbose)
{
    quint16 prevGoodValue = videoParameters.black16bIre;
    bool forceDropout = false;
    QVector<QVector<quint16>> tmpField(videoParameters.fieldHeight * videoParameters.fieldWidth);

    // Optionally phase-align each source's chroma subcarrier to a reference
    // source (first available) before combining, so that sub-sample subcarrier
    // phase wander between sources does not cancel chroma on averaging.  Work on
    // a local copy so dropout metadata (which indexes the original layout) is
    // unaffected -- only sample VALUES change.
    QVector<SourceVideo::Data> alignedStorage;
    const QVector<SourceVideo::Data>* fieldsPtr = &inputFields;
    if (chromaAlign && availableSourcesForFrame.size() > 1) {
        alignedStorage = inputFields;
        alignChroma(alignedStorage, availableSourcesForFrame, videoParameters);
        fieldsPtr = &alignedStorage;
    }
    const QVector<SourceVideo::Data>& inputFieldsV = *fieldsPtr;

    if (availableSourcesForFrame.size() > 0) {
        // Sources available - process field
        for (qint32 y = 0; y < videoParameters.fieldHeight; y++) {
            for (qint32 x = 0; x < videoParameters.fieldWidth; x++) {
                QVector<quint16> valuesN;//North neighbor pixel
                QVector<quint16> valuesS;//South neighbor pixel
                QVector<quint16> valuesE;//East neighbor pixel
                QVector<quint16> valuesW;//West neighbor pixel
                QVector<bool> isAllDropout = {true,true,true,true,true};//is neighbor pixel all dropout : current = [0] / N = [1] / S = [2] / E = [3] / W = [4]

                QVector<quint16> inputValues;
                // Get input values from the input sources (which are not marked as dropouts)
                if(mode >= 3)//get surrounding pixels
                {
                    Stacker::getProcessedSample(x, y, availableSourcesForFrame, inputFieldsV, tmpField, videoParameters, fieldMetadata, inputValues, valuesN, valuesS, valuesE, valuesW, isAllDropout, noDiffDod, verbose);
                }
                else// get only pixel 1 by 1
                {
                    for (qint32 i = 0; i < availableSourcesForFrame.size(); i++){
                        //read pixel
                        const quint16 pixelValue = inputFieldsV[availableSourcesForFrame[i]][(videoParameters.fieldWidth * y) + x];
                        const bool sampleIsDropout = isDropout(fieldMetadata[availableSourcesForFrame[i]].dropOuts, x, y);

                        // Include the source's pixel data if it's not marked as a dropout
                        if (!sampleIsDropout) {
                            // Pixel is valid
                            inputValues.append(pixelValue);
                        }
                        else if((pixelValue > 0) && (!noDiffDod))
                        {
                            inputValues.append(pixelValue);
                        }

                        if(!sampleIsDropout)
                        {
                            isAllDropout[0] = false;
                        }
                    }

                    // If all possible input values are dropouts (and noDiffDod is false) and there are more than 3 input sources...
                    // Take the available values (marked as dropouts) and perform a diffDOD to try and determine if the dropout markings
                    // are false positives.
                    if (isAllDropout[0] && (availableSourcesForFrame.size() >= 3) && !noDiffDod) {
                        // Perform differential dropout detection to recover ld-decode false positive pixels
                        if(x > videoParameters.colourBurstStart)
                        {
                            inputValues = diffDod(inputValues, videoParameters, verbose);

                            if(verbose)
                            {
                                if (inputValues.size() > 0) {
                                    qInfo().nospace() << "Frame #" << frameNumber << ": DiffDOD recovered " << inputValues.size() <<
                                                         " values: " << inputValues << " for field location (" << x << ", " << y << ")";
                                } else if(x > videoParameters.colourBurstStart){
                                    qInfo().nospace() << "Frame #" << frameNumber << ": DiffDOD failed, no values recovered for field location (" << x << ", " << y << ")";
                                }
                                else{
                                    qInfo().nospace() << "Frame #" << frameNumber << ": Values 0 recovered for field location (" << x << ", " << y << ")";
                                }
                            }
                        }
                    }
                }

                // If passThrough is set, the output is always marked as a dropout if all input values are dropouts
                // (regardless of the diffDOD process result).
                forceDropout = false;
                if ((availableSourcesForFrame.size() > 0) && (passThrough)) {
                    if(x > videoParameters.colourBurstStart)
                    {
                        if (inputValues.size() == 0) {
                            forceDropout = true;
                            if(verbose)
                            {
                                qInfo().nospace() << "Frame #" << frameNumber << ": All sources for field location (" << x << ", " << y << ") are marked as dropout, passing through";
                            }
                        }
                    }
                }

                // Stack with intelligence:
                // If there are 3 or more sources - median (with central average for non-odd source sets)
                // If there are 2 sources - average
                // If there is 1 source - output as is
                // If there are zero sources - mark as a dropout in the output file
                if (inputValues.size() == 0) {
                    // No values available - use the previous good value and mark as a dropout
                    outputField[(videoParameters.fieldWidth * y) + x] = prevGoodValue;
                    if(x > videoParameters.colourBurstStart){dropOuts.append(x, x, y + 1);}
                } else if (inputValues.size() == 1) {
                    // 1 value available - just copy it to the output
                    outputField[(videoParameters.fieldWidth * y) + x] = inputValues[0];
                    prevGoodValue = outputField[(videoParameters.fieldWidth * y) + x];
                    if (forceDropout) dropOuts.append(x, x, y + 1);
                } else {
                    //2 or more values available - store the result in the output field
                    outputField[(videoParameters.fieldWidth * y) + x] = stackMode(inputValues, valuesN, valuesS, valuesE, valuesW, isAllDropout, mode, smartThreshold);
                    prevGoodValue = outputField[(videoParameters.fieldWidth * y) + x];
                    tmpField[(videoParameters.fieldWidth * y) + x] = QVector<quint16>{prevGoodValue};
                    if (forceDropout) dropOuts.append(x, x, y + 1);
                }
            }
        }

        // Concatenate the dropouts
        if (dropOuts.size() != 0) dropOuts.concatenate(verbose);
    } else {
        // No sources available for field - generate a dummy field at the black IRE level
        for (qint32 y = 0; y < videoParameters.fieldHeight; y++) {
            for (qint32 x = videoParameters.colourBurstStart; x < videoParameters.fieldWidth; x++) {
                outputField[(videoParameters.fieldWidth * y) + x] = videoParameters.black16bIre;
            }
        }
    }
}

// Phase-align each non-reference source's chroma subcarrier (fSC = fs/4) to the
// reference source, per line, over the active region.  Composite sources of the
// same disc carry a slightly different subcarrier phase (sub-sample time-base
// wander); averaging them sample-by-sample then partially CANCELS chroma.  Here
// each source's chroma band is demodulated to baseband I/Q (quadrature mix at
// fs/4 + low-pass), its bulk phase offset to the reference is measured, the I/Q
// is rotated to match, and the chroma is re-modulated and written back -- luma
// (outside the band) is left untouched.  Mirrors lddecode/stack.py
// chroma_align_field, done without an FFT (4xfSC sampling makes the I/Q mix the
// fixed sequences cos=[1,0,-1,0], sin=[0,1,0,-1]).
void Stacker::alignChroma(QVector<SourceVideo::Data>& fields,
                          const QVector<qint32>& availableSourcesForFrame,
                          const LdDecodeMetaData::VideoParameters& videoParameters)
{
    const qint32 nSrc = availableSourcesForFrame.size();
    if (nSrc < 2) return;
    const qint32 W = videoParameters.fieldWidth;
    const qint32 H = videoParameters.fieldHeight;
    qint32 x0 = videoParameters.activeVideoStart;
    qint32 x1 = videoParameters.activeVideoEnd;
    if (x0 < 0) x0 = 0;
    if (x1 > W) x1 = W;
    const qint32 n = x1 - x0;
    if (n < 16) return;

    // fs/4 quadrature carriers, indexed by absolute column x & 3
    static const double cosv[4] = {1.0, 0.0, -1.0, 0.0};
    static const double sinv[4] = {0.0, 1.0, 0.0, -1.0};

    // low-pass kernel (Hamming-windowed sinc, cutoff ~0.12) to isolate the
    // baseband chroma after quadrature mixing (rejects the fs/4 and fs/2 terms)
    static QVector<double> h;
    if (h.isEmpty()) {
        const double PI = 3.14159265358979323846;
        // narrow baseband cutoff (~0.06) to match the +-0.06 chroma half-band
        // around fSC: over a narrow band the source-to-reference offset is ~a
        // constant phase, so a single per-line rotation corrects it well.
        const int L = 23; const int c = L / 2; const double fc = 0.06;
        h.resize(L); double s = 0.0;
        for (int k = 0; k < L; ++k) {
            const double m = k - c;
            const double sinc = (m == 0.0) ? (2.0 * fc)
                                           : std::sin(2.0 * PI * fc * m) / (PI * m);
            const double win = 0.54 - 0.46 * std::cos(2.0 * PI * k / (L - 1));
            h[k] = sinc * win; s += h[k];
        }
        for (int k = 0; k < L; ++k) h[k] /= s;
    }
    const int L = h.size(); const int c = L / 2;

    QVector<double> Iraw(n), Qraw(n), Ir(n), Qr(n), Is(n), Qs(n);
    auto demod = [&](const SourceVideo::Data& fld, qint32 y,
                     QVector<double>& I, QVector<double>& Q) {
        for (qint32 i = 0; i < n; ++i) {
            const qint32 x = x0 + i;
            const double v = static_cast<double>(fld[(W * y) + x]);
            Iraw[i] = v * cosv[x & 3];
            Qraw[i] = -v * sinv[x & 3];
        }
        for (qint32 i = 0; i < n; ++i) {
            double si = 0.0, sq = 0.0;
            for (int k = 0; k < L; ++k) {
                qint32 j = i + k - c;
                if (j < 0) j = 0; else if (j >= n) j = n - 1;
                si += h[k] * Iraw[j];
                sq += h[k] * Qraw[j];
            }
            I[i] = 2.0 * si; Q[i] = 2.0 * sq;
        }
    };

    const qint32 refSrc = availableSourcesForFrame[0];
    for (qint32 y = 0; y < H; ++y) {
        demod(fields[refSrc], y, Ir, Qr);
        for (qint32 si = 1; si < nSrc; ++si) {
            const qint32 s = availableSourcesForFrame[si];
            demod(fields[s], y, Is, Qs);
            // bulk phase offset of this source's chroma vs the reference's
            double re = 0.0, im = 0.0, magS = 0.0, magR = 0.0;
            for (qint32 i = 0; i < n; ++i) {
                re += Is[i] * Ir[i] + Qs[i] * Qr[i];
                im += Qs[i] * Ir[i] - Is[i] * Qr[i];
                magS += Is[i] * Is[i] + Qs[i] * Qs[i];
                magR += Ir[i] * Ir[i] + Qr[i] * Qr[i];
            }
            const double denom = std::sqrt(magS * magR);
            const double mag = std::sqrt(re * re + im * im);
            // skip lines with little/no chroma in either source (phase is noise)
            if (denom < 1e-9 || mag < 0.05 * denom) continue;
            const double dphi = std::atan2(im, re);
            const double cd = std::cos(dphi), sd = std::sin(dphi);
            SourceVideo::Data& line = fields[s];
            for (qint32 i = 0; i < n; ++i) {
                const qint32 x = x0 + i;
                const double Ip = Is[i] * cd + Qs[i] * sd;   // rotate by -dphi
                const double Qp = -Is[i] * sd + Qs[i] * cd;
                const double cRecon = Is[i] * cosv[x & 3] - Qs[i] * sinv[x & 3];
                const double cAlign = Ip * cosv[x & 3]     - Qp * sinv[x & 3];
                double out = static_cast<double>(line[(W * y) + x]) - cRecon + cAlign;
                if (out < 0.0) out = 0.0; else if (out > 65535.0) out = 65535.0;
                line[(W * y) + x] = static_cast<quint16>(out + 0.5);
            }
        }
    }
}

// Method to stack a vector of quint16 using a selected mode
quint16 Stacker::stackMode(const QVector<quint16>& elements, const QVector<quint16>& elementsN, const QVector<quint16>& elementsS, const QVector<quint16>& elementsE, const QVector<quint16>& elementsW, const QVector<bool>& isAllDropout, const qint32& mode, const qint32& smartThreshold)
{
    const qint32 nbOfElements = elements.size();
    qint32 nbSelected = 0;
    quint32 result = 0;
    QVector<quint16> closestList;

    //neighbor pixel
    qint32 resultN = 0;
    qint32 resultS = 0;
    qint32 resultE = 0;
    qint32 resultW = 0;
    quint32 resultNeighbor = 0;

    qint32 nbNeighbor = 0;

    switch (mode) {
        case 0://mean mode
        {
            result = Stacker::mean(elements);
            break;
        }
        case 1://median mode
        {
            result = Stacker::median(elements);
            break;
        }
        case 2://smart mean mode
        {
            const qint32 median = Stacker::median(elements);
            //count number of sample within threshold distance to the median and sum
            for(int i=0; i < nbOfElements;i++)
            {
                if(elements[i] < (median + smartThreshold) &&  elements[i] > (median - smartThreshold))
                {
                    nbSelected++;
                    result += elements[i];
                }
            }
            //select median if all other source are out of the threshold range
            if(nbSelected == 0)
            {
                result = median;
            }
            else//mean averaging of selected sample
            {
                result = (result / nbSelected);
            }
            break;
        }
        case 3://smart neighbor mode
        {
            const qint32 median = Stacker::median(elements);

            ((elementsN.size() > 1) && isAllDropout[1]) ? resultN = Stacker::median(elementsN) : (elementsN.size() > 0 ? resultN = elementsN[0] : resultN = -1);
            ((elementsS.size() > 1) && isAllDropout[2]) ? resultS = Stacker::median(elementsS) : (elementsS.size() > 0 ? resultS = elementsS[0] : resultS = -1);

            if(!isAllDropout[0])
            {
                ((elementsE.size() > 1) && isAllDropout[3]) ? resultE = Stacker::median(elementsE) : (elementsE.size() > 0 ? resultE = elementsE[0] : resultE = -1);
                ((elementsW.size() > 1) && isAllDropout[4]) ? resultW = Stacker::median(elementsW) : (elementsW.size() > 0 ? resultW = elementsW[0] : resultW = -1);
            }

            //check number of neighbor available and prepare for mean
            (resultN != -1) ? nbNeighbor++ : resultN = 0;
            (resultS != -1) ? nbNeighbor++ : resultS = 0;
            (resultE != -1) ? nbNeighbor++ : resultE = 0;
            (resultW != -1) ? nbNeighbor++ : resultW = 0;

            if(nbNeighbor > 0)
            {
                //closest value to a neighbor
                if(resultN > 0){closestList.append(Stacker::closest(elements, resultN));}
                if(resultS > 0){closestList.append(Stacker::closest(elements, resultS));}
                if(resultE > 0){closestList.append(Stacker::closest(elements, resultE));}
                if(resultW > 0){closestList.append(Stacker::closest(elements, resultW));}

                resultNeighbor = Stacker::closest(closestList, median);//get the closest value to the median/mean based on closest value to a neighbor
            }
            else
            {
                resultNeighbor = result;
            }

            if(nbOfElements > 2)//using median + mean
            {
                result = 0;
                //count number of sample within threshold distance to the median and sum
                for(int i=0; i < nbOfElements;i++)
                {
                    if((elements[i] < (resultNeighbor + smartThreshold)) && (elements[i] > (resultNeighbor - smartThreshold)))
                    {
                        nbSelected++;
                        result += elements[i];
                    }
                }

                //select median if all other source are out of the threshold range
                if(nbSelected == 0)
                {
                    result = resultNeighbor;
                }
                //mean averaging of selected sample
                else
                {
                    result = (result / nbSelected);
                }
            }
            else//using surrounding sample
            {
                result = resultNeighbor;// get the the closest value to neighbor
            }
            break;
        }
        case 4://neighbor mode
        {
            const qint32 median = Stacker::median(elements);

            ((elementsN.size() > 1) && isAllDropout[1]) ? resultN = Stacker::median(elementsN) : (elementsN.size() > 0 ? resultN = elementsN[0] : resultN = -1);
            ((elementsS.size() > 1) && isAllDropout[2]) ? resultS = Stacker::median(elementsS) : (elementsS.size() > 0 ? resultS = elementsS[0] : resultS = -1);

            if(!isAllDropout[0] || (isAllDropout[1] && isAllDropout[2]))
            {
                ((elementsE.size() > 1) && isAllDropout[3]) ? resultE = Stacker::median(elementsE) : (elementsE.size() > 0 ? resultE = elementsE[0] : resultE = -1);
                ((elementsW.size() > 1) && isAllDropout[4]) ? resultW = Stacker::median(elementsW) : (elementsW.size() > 0 ? resultW = elementsW[0] : resultW = -1);
            }


            //check number of neighbor available and prepare for mean
            (resultN != -1) ? nbNeighbor++ : resultN = 0;
            (resultS != -1) ? nbNeighbor++ : resultS = 0;
            (resultE != -1) ? nbNeighbor++ : resultE = 0;
            (resultW != -1) ? nbNeighbor++ : resultW = 0;

            if(nbNeighbor > 0)
            {
                if(resultN > 0){closestList.append(Stacker::closest(elements, resultN));}
                if(resultS > 0){closestList.append(Stacker::closest(elements, resultS));}
                if(resultE > 0){closestList.append(Stacker::closest(elements, resultE));}
                if(resultW > 0){closestList.append(Stacker::closest(elements, resultW));}

                result = Stacker::closest(closestList, median);//get the closest value to the median/mean based on closest value to a neighbor

                if(nbOfElements > 2)
                {
                    result = (median + result) / 2;// get the mean between (median/mean) and the closest value to neighbor
                }
            }
            else
            {
                result = median;
            }
            break;
        }
    }

    return static_cast<quint16>(result);
}

// Method to find the median of a vector of quint16s
inline quint16 Stacker::median(QVector<quint16> elements)
{
    const qint32 noOfElements = elements.size();

    if (noOfElements % 2 == 0) {
        // Input set is even length

        // Applying nth_element on n/2th index
        std::nth_element(elements.begin(), elements.begin() + noOfElements / 2, elements.end());

        // Applying nth_element on (n-1)/2 th index
        std::nth_element(elements.begin(), elements.begin() + (noOfElements - 1) / 2, elements.end());

        // Find the average of value at index N/2 and (N-1)/2
        return static_cast<quint16>((elements[(noOfElements - 1) / 2] + elements[noOfElements / 2]) / 2.0);
    } else {
        // Input set is odd length

        // Applying nth_element on n/2
        std::nth_element(elements.begin(), elements.begin() + noOfElements / 2, elements.end());

        // Value at index (N/2)th is the median
        return static_cast<quint16>(elements[noOfElements / 2]);
    }
}

// Method to find the mean of a vector of quint16s
inline qint32 Stacker::mean(const QVector<quint16>& elements)
{
    quint32 result = 0;
    const qint32 nbElements = elements.size();

    if(nbElements > 1)
    {
        //compute mean of all values
        for(int i=0; i < nbElements;i++)
        {
            if(nbElements > 1)
            {
                result += elements[i];
            }
        }
        return (result / nbElements);
    }
    else if(nbElements == 1)
    {
        return elements[0];
    }
    else
    {
        return -1;
    }

}

// Method to find the closest value to a target
inline quint16 Stacker::closest(const QVector<quint16>& elements, const qint32 target)
{
    const qint32 nbOfElements = elements.size();
    qint32 closest = 0;

    if(nbOfElements > 0)
    {
        closest = elements[0];
        for(int i=1;i < nbOfElements;i++)
        {
            if(abs(target - elements[i]) < abs(target - closest))
            {
                closest = elements[i];
            }
        }
    }

    return closest;
}

// get value that are unprocessed and reuse processed one for mode >= 3
void Stacker::getProcessedSample(const qint32 x, const qint32 y, const QVector<qint32>& availableSourcesForFrame, const QVector<SourceVideo::Data>& inputFields, QVector<QVector<quint16>>& tmpField, const LdDecodeMetaData::VideoParameters& videoParameters, const QVector<LdDecodeMetaData::Field>& fieldMetadata, QVector<quint16>& sample, QVector<quint16>& sampleN, QVector<quint16>& sampleS, QVector<quint16>& sampleE, QVector<quint16>& sampleW, QVector<bool>& isAllDropout, const bool& noDiffDod, const bool& verbose)
{
    quint16 pixelValue = 0;
    qint32 source = 0;
    qint32 fieldWidth = videoParameters.fieldWidth;
    qint32 fieldHeight = videoParameters.fieldHeight;
    bool sampleIsDropout = true;
    for (qint32 i = 0; i < availableSourcesForFrame.size(); i++) {
        source = availableSourcesForFrame[i];
        if(y == 0)
        {
            if(x == 0)//read value + east + south
            {
                //read new value
                pixelValue = inputFields[source][(fieldWidth * y) + x];
                sampleIsDropout = isDropout(fieldMetadata[source].dropOuts, x, y);
                if (!sampleIsDropout) {
                    // Pixel is valid
                    sample.append(pixelValue);
                }
                else if((pixelValue > 0) && (!noDiffDod))
                {
                    sample.append(pixelValue);
                }

                if(!sampleIsDropout)
                {
                    isAllDropout[0] = false;
                }

                pixelValue = inputFields[source][(fieldWidth * y) + x + 1];
                sampleIsDropout = isDropout(fieldMetadata[source].dropOuts, x+1, y);
                if (!sampleIsDropout) {
                    // Pixel is valid
                    sampleE.append(pixelValue);
                }
                else if((pixelValue > 0) && (!noDiffDod))
                {
                    sampleE.append(pixelValue);
                }

                if(!sampleIsDropout)
                {
                    isAllDropout[3] = false;//E = [3]
                }

                pixelValue = inputFields[source][(fieldWidth * (y+1)) + x];
                sampleIsDropout = isDropout(fieldMetadata[source].dropOuts, x, y+1);
                if (!sampleIsDropout) {
                    // Pixel is valid
                    sampleS.append(pixelValue);
                }
                else if((pixelValue > 0) && (!noDiffDod))
                {
                    sampleS.append(pixelValue);
                }

                if(!sampleIsDropout)
                {
                    isAllDropout[2] = false;//S = [2]
                }
            }
            else if(x == fieldWidth -1)//read south value
            {
                //read new value
                pixelValue = inputFields[source][(fieldWidth * (y+1)) + x];
                sampleIsDropout = isDropout(fieldMetadata[source].dropOuts, x, y+1);
                if (!sampleIsDropout) {
                    // Pixel is valid
                    sampleS.append(pixelValue);
                }
                else if((pixelValue > 0) && (!noDiffDod))
                {
                    sampleS.append(pixelValue);
                }

                if(!sampleIsDropout)
                {
                    isAllDropout[2] = false;//S = [2]
                }
            }
            else//read east + south
            {
                //read new value
                pixelValue = inputFields[source][(fieldWidth * y) + x + 1];
                sampleIsDropout = isDropout(fieldMetadata[source].dropOuts, x+1, y);
                if (!sampleIsDropout) {
                    // Pixel is valid
                    sampleE.append(pixelValue);
                }
                else if((pixelValue > 0) && (!noDiffDod))
                {
                    sampleE.append(pixelValue);
                }

                if(!sampleIsDropout)
                {
                    isAllDropout[3] = false;//E = [3]
                }

                pixelValue = inputFields[source][(fieldWidth * (y+1)) + x];
                sampleIsDropout = isDropout(fieldMetadata[source].dropOuts, x, y+1);
                if (!sampleIsDropout) {
                    // Pixel is valid
                    sampleS.append(pixelValue);
                }
                else if((pixelValue > 0) && (!noDiffDod))
                {
                    sampleS.append(pixelValue);
                }

                if(!sampleIsDropout)
                {
                    isAllDropout[2] = false;//S = [2]
                }
            }
        }
        else if(y != fieldHeight -1)//read south value
        {
            if(x == 0)//get neighbor value except on left
            {
                //read new value
                pixelValue = inputFields[source][(fieldWidth * (y+1)) + x];
                sampleIsDropout = isDropout(fieldMetadata[source].dropOuts, x, y+1);
                if (!sampleIsDropout) {
                    // Pixel is valid
                    sampleS.append(pixelValue);
                }
                else if((pixelValue > 0) && (!noDiffDod))
                {
                    sampleS.append(pixelValue);
                }

                if(!sampleIsDropout)
                {
                    isAllDropout[2] = false;//S = [2]
                }
            }
            if(x == fieldWidth -1)//get neighbor value except on right
            {
                //read new value
                pixelValue = inputFields[source][(fieldWidth * (y+1)) + x];
                sampleIsDropout = isDropout(fieldMetadata[source].dropOuts, x, y+1);
                if (!sampleIsDropout) {
                    // Pixel is valid
                    sampleS.append(pixelValue);
                }
                else if((pixelValue > 0) && (!noDiffDod))
                {
                    sampleS.append(pixelValue);
                }

                if(!sampleIsDropout)
                {
                    isAllDropout[2] = false;//S = [2]
                }
            }
            else
            {
                //read new value
                pixelValue = inputFields[source][(fieldWidth * (y+1)) + x];
                sampleIsDropout = isDropout(fieldMetadata[source].dropOuts, x, y+1);
                if (!sampleIsDropout) {
                    // Pixel is valid
                    sampleS.append(pixelValue);
                }
                else if((pixelValue > 0) && (!noDiffDod))
                {
                    sampleS.append(pixelValue);
                }

                if(!sampleIsDropout)
                {
                    isAllDropout[2] = false;//S = [2]
                }
            }
        }
    }
    // If all possible input values are dropouts (and noDiffDod is false) and there are more than 3 input sources...
    // Take the available values (marked as dropouts) and perform a diffDOD to try and determine if the dropout markings
    // are false positives.
    if(y == 0)
    {
        if(x == 0)//read value + east + south
        {
            if(!noDiffDod)
            {
                if(x > videoParameters.colourBurstStart)
                {
                    if (isAllDropout[0] && (availableSourcesForFrame.size() >= 3)) {
                        sample = diffDod(sample, videoParameters, verbose);
                    }
                    if (isAllDropout[3] && (availableSourcesForFrame.size() >= 3)) {
                        sampleE = diffDod(sampleE, videoParameters, verbose);
                    }
                    if (isAllDropout[2] && (availableSourcesForFrame.size() >= 3)) {
                        sampleS = diffDod(sampleS, videoParameters, verbose);
                    }
                }
            }
            tmpField[(fieldWidth * y) + x] = sample;
            tmpField[(fieldWidth * y) + x + 1] = sampleE;
            tmpField[(fieldWidth * (y+1)) + x] = sampleS;
        }
        else if(x == fieldWidth -1)//read south value
        {
            if(!noDiffDod)
            {
                if(x > videoParameters.colourBurstStart)
                {
                    if (isAllDropout[2] && (availableSourcesForFrame.size() >= 3)) {
                        sampleS = diffDod(sampleS, videoParameters, verbose);
                    }
                }
            }
            tmpField[(fieldWidth * (y+1)) + x] = sampleS;
            sample = tmpField[(fieldWidth * y) + x];
            sampleW = tmpField[(fieldWidth * y) + x - 1];
            isAllDropout[4] = haveAllDropout(fieldMetadata,x-1,y);
        }
        else//read east + south
        {
            if(!noDiffDod)
            {
                if(x > videoParameters.colourBurstStart)
                {
                    if (isAllDropout[3] && (availableSourcesForFrame.size() >= 3)) {
                        sampleE = diffDod(sampleE, videoParameters, verbose);
                    }
                    if (isAllDropout[2] && (availableSourcesForFrame.size() >= 3)) {
                        sampleS = diffDod(sampleS, videoParameters, verbose);
                    }
                }
            }
            tmpField[(fieldWidth * y) + x + 1] = sampleE;
            tmpField[(fieldWidth * (y+1)) + x] = sampleS;
            sample = tmpField[(fieldWidth * y) + x];
            sampleW = tmpField[(fieldWidth * y) + x - 1];
            isAllDropout[4] = haveAllDropout(fieldMetadata,x-1,y);
        }
    }
    else if(y != fieldHeight -1)//read south value
    {
        if(!noDiffDod)
        {
            if(x > videoParameters.colourBurstStart)
            {
                if (isAllDropout[2] && (availableSourcesForFrame.size() >= 3)) {
                    sampleS = diffDod(sampleS, videoParameters, verbose);
                }
            }
        }
        tmpField[(fieldWidth * (y+1)) + x] = sampleS;
        if(x == 0)
        {
            sample = tmpField[(fieldWidth * y) + x];
            sampleE = tmpField[(fieldWidth * y) + x + 1];
            sampleN = tmpField[(fieldWidth * (y-1)) + x];
            isAllDropout[1] = haveAllDropout(fieldMetadata,x,y-1);
            isAllDropout[3] = haveAllDropout(fieldMetadata,x+1,y);
        }
        else if (x == fieldWidth -1)
        {
            sample = tmpField[(fieldWidth * y) + x];
            sampleW = tmpField[(fieldWidth * y) + x - 1];
            sampleN = tmpField[(fieldWidth * (y-1)) + x];
            isAllDropout[1] = haveAllDropout(fieldMetadata,x,y-1);
            isAllDropout[4] = haveAllDropout(fieldMetadata,x-1,y);
        }
        else
        {
            sample = tmpField[(fieldWidth * y) + x];
            sampleW = tmpField[(fieldWidth * y) + x - 1];
            sampleE = tmpField[(fieldWidth * y) + x + 1];
            sampleN = tmpField[(fieldWidth * (y-1)) + x];
            isAllDropout[1] = haveAllDropout(fieldMetadata,x,y-1);
            isAllDropout[3] = haveAllDropout(fieldMetadata,x+1,y);
            isAllDropout[4] = haveAllDropout(fieldMetadata,x-1,y);
        }
    }
    else//all value already processed : reuse value
    {
        if(x == 0)
        {
            sample = tmpField[(fieldWidth * y) + x];
            sampleE = tmpField[(fieldWidth * y) + x + 1];
            sampleN = tmpField[(fieldWidth * (y-1)) + x];
            isAllDropout[1] = haveAllDropout(fieldMetadata,x,y-1);
            isAllDropout[3] = haveAllDropout(fieldMetadata,x+1,y);
        }
        if(x == fieldWidth -1)
        {
            sample = tmpField[(fieldWidth * y) + x];
            sampleW = tmpField[(fieldWidth * y) + x - 1];
            sampleN = tmpField[(fieldWidth * (y-1)) + x];
            isAllDropout[1] = haveAllDropout(fieldMetadata,x,y-1);
            isAllDropout[4] = haveAllDropout(fieldMetadata,x-1,y);
        }
        else
        {
            sample = tmpField[(fieldWidth * y) + x];
            sampleW = tmpField[(fieldWidth * y) + x - 1];
            sampleE = tmpField[(fieldWidth * y) + x + 1];
            sampleN = tmpField[(fieldWidth * (y-1)) + x];
            isAllDropout[1] = haveAllDropout(fieldMetadata,x,y-1);
            isAllDropout[3] = haveAllDropout(fieldMetadata,x+1,y);
            isAllDropout[4] = haveAllDropout(fieldMetadata,x-1,y);
        }
    }
}

// Method returns true if specified pixel is a dropout
inline bool Stacker::isDropout(const DropOuts& dropOuts, const qint32 fieldX, const qint32 fieldY)
{
    for (qint32 i = 0; i < dropOuts.size(); i++) {
        if ((dropOuts.fieldLine(i) - 1) == fieldY) {
            if ((fieldX >= dropOuts.startx(i)) && (fieldX <= dropOuts.endx(i)))
                return true;
        }
    }

    return false;
}

// Method returns true if all specified pixel are dropouts
inline bool Stacker::haveAllDropout(const QVector<LdDecodeMetaData::Field>& fieldMetadata, const qint32 x, const qint32 y)
{
    const qint32 size = fieldMetadata.size();
    for (qint32 i = 0; i < size; i++) {
        if(!isDropout(fieldMetadata[i].dropOuts,x,y))
            return false;
    }

    return true;
}

// Use differential dropout detection to remove suspected dropout error
// values from inputValues to produce the set of output values.  This generally improves everything, but
// might cause an increase in errors for really noisy frames (where the DOs are in the same place in
// multiple sources).  Another possible disadvantage is that diffDOD might pass through master plate errors
// which, whilst not technically errors, may be undesirable.
QVector<quint16> Stacker::diffDod(const QVector<quint16>& inputValues, const LdDecodeMetaData::VideoParameters& videoParameters, const bool& verbose)
{
    QVector<quint16> outputValues;

    // Check that we have at least 3 input values
    if (inputValues.size() < 3) {
        return inputValues;
    }

    // Get the median value of the input values
    const double medianValue = static_cast<double>(median(inputValues));

    // Set the matching threshold to +-10% of the median value
    const double threshold = 10; // %

    // Set the maximum and minimum values for valid inputs
    double maxValueD = medianValue + ((medianValue / 100.0) * threshold);
    double minValueD = medianValue - ((medianValue / 100.0) * threshold);
    if (minValueD < 0) minValueD = 0;
    if (maxValueD > 65535) maxValueD = 65535;
    quint16 minValue = minValueD;
    quint16 maxValue = maxValueD;

    // Copy valid input values to the output set
    for (qint32 i = 0; i < inputValues.size(); i++) {
        if ((inputValues[i] > minValue) && (inputValues[i] < maxValue)) {
            outputValues.append(inputValues[i]);
        }
    }

    // Show debug
    if(verbose)
    {
        qDebug() << "diffDOD:  Input" << inputValues;
        if (outputValues.size() == 0) {
            qDebug().nospace() << "diffDOD: Empty output... Range was " << minValue << "-" << maxValue << " with a median of " << medianValue;
        } else {
            qDebug() << "diffDOD: Output" << outputValues;
        }
    }

    return outputValues;
}
