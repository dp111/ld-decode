/************************************************************************

    adfs_verifier.h

    vfs-verifier - Acorn VFS (Domesday) image verifier
    Copyright (C) 2025 Simon Inns

    This file is part of ld-decode-tools.

    This application is free software: you can redistribute it and/or
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

#ifndef ADFS_VERIFIER_H
#define ADFS_VERIFIER_H

#include <QString>
#include <QDebug>
#include <QFile>

#include "adfs_image.h"
#include "adfs_fsm.h"
#include "adfs_directory.h"
#include "adfs_filecheck.h"
#include "bad_sectors.h"

class AdfsVerifier
{
public:
    AdfsVerifier();
    
    bool process(const QString &filename, const QString &bsmFilename);

private:
    AdfsImage m_image;
    void hexDump(QByteArray &data, qint32 startSector) const;
};

#endif // ADFS_VERIFIER_H