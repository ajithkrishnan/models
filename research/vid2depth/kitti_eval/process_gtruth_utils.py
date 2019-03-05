import numpy as np
from pose_evaluation_utils import rot2quat

#DEBUG
#def quat2rot(quaternion):
#    """Return homogeneous rotation matrix from quaternion.
#
#    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
#    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
#    True
#    >>> M = quaternion_matrix([1, 0, 0, 0])
#    >>> numpy.allclose(M, numpy.identity(4))
#    True
#    >>> M = quaternion_matrix([0, 1, 0, 0])
#    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
#    True
#
#    """
#    q = np.array(quaternion, dtype=np.float64, copy=True)
#    n = np.dot(q, q)
##    if n < _EPS:
##        return np.identity(4)
#    q *= math.sqrt(2.0 / n)
#    q = np.outer(q, q)
#    return np.array([
#        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
#        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
#        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])


def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def pose_vec_to_mat(vec):
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3,1))
#    rot = quat2mat([vec[6], vec[5], vec[4], vec[3]])
    rot = np.array(
            [[ vec[0], vec[1], vec[2] ],
             [ vec[4], vec[5], vec[6] ],
             [ vec[8], vec[9], vec[10] ]]).reshape((3,3))
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat

def dump_pose_seq_TUM(out_file, poses, times):
    # Set first frame as the origin
    first_origin = pose_vec_to_mat(poses[0])
    with open(out_file, 'w') as f:
        for p in range(len(times)):
            this_pose = pose_vec_to_mat(poses[p])
            this_pose = np.dot(first_origin, np.linalg.inv(this_pose))
            tx = this_pose[0, 3]
            ty = this_pose[1, 3]
            tz = this_pose[2, 3]
            rot = this_pose[:3, :3]
            qw, qx, qy, qz = rot2quat(rot)
            f.write('%f %f %f %f %f %f %f %f\n' % (times[p], tx, ty, tz, qx, qy, qz, qw))

def load_sequence(dataset_dir, 
                        tgt_idx, 
                        gt_array, 
                        seq_length):
#    max_offset = int((seq_length - 1)/2)
    max_offset = 1
#    for o in range(-max_offset, max_offset+1):
    for o in range(0, max_offset + 2):
        curr_idx = tgt_idx + o
        curr_pose = gt_array[curr_idx]
#        if o == -max_offset:
        if o == 0:
            pose_seq = curr_pose 
        else:
            pose_seq = np.vstack((pose_seq, curr_pose))
    return pose_seq


def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)

    if tgt_idx >= N:
      return False
    tgt_drive, _ = frames[tgt_idx].split(' ')
    #TODO: calculate max_offset in a clean way 
#    max_offset = (seq_length - 1)//2
    max_offset = 1
#    min_src_idx = tgt_idx - max_offset
#    max_src_idx = tgt_idx + max_offset
    min_src_idx = tgt_idx 
    max_src_idx = tgt_idx + 2*max_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

