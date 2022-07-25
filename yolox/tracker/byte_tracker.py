import numpy as np

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30, det_thresh=[0.1, 0.6]):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh_low = det_thresh[0]   # low detection score threshold
        self.det_thresh_high = det_thresh[1]  # high detection score threshold
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        img_h, img_w = img_info[0], img_info[1]

        if len(img_size) > 0:
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        """ Step 1: Split detections into high score detections and low score detections"""
        # In the official repo, `self.args.track_thresh` (default: 0.6) is used to obtain high/low score detections
        inds_high = scores >= self.det_thresh_high
        inds_low = (~inds_high) & (scores > self.det_thresh_low)

        """ Step 2.1: First association between activated tracks and high score detections"""
        # Initialize new tracklets with high score detections in the current frame
        dets_high = [
            STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(bboxes[inds_high], scores[inds_high])
        ] if inds_high.any() else []

        tracked_stracks = [
            track for track in self.tracked_stracks if track.is_activated
        ]  # type: list[STrack]
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict new locations of activated tracks in the current frame with Kalman Filter
        STrack.multi_predict(strack_pool)

        # Matching
        dists = matching.iou_distance(strack_pool, dets_high)
        # TODO: Why fusing high detction score into matching distance?
        if not self.args.mot20:
            dists = matching.fuse_score(dists, dets_high)
        # if the IoU between a matching is smaller than `1 - thresh`, then reject the matching
        matches, u_tracks, u_dets_high = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # Update tracks matched successfully
        for itrack, idet in matches:
            track = strack_pool[itrack]
            if track.state == TrackState.Tracked:
                track.update(dets_high[idet], self.frame_id)
                activated_starcks.append(track)
            else:  # state == TrackState.Lost
                track.re_activate(dets_high[idet], self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 2.3: Second association between tracked tracks unmatched above and low score detections"""
        # Initialize new tracklets with low score detections in the current frame
        dets_low = [
            STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(bboxes[inds_low], scores[inds_low])
        ] if inds_low.any() else []

        # Activated tracks matched in last frame but unmatched in current frame's first association (Step 2.1)
        r_tracked_stracks = [strack_pool[i] for i in u_tracks if strack_pool[i].state == TrackState.Tracked]

        # Matching  # TODO: Why not fusing here? (because detection score is low?)
        dists = matching.iou_distance(r_tracked_stracks, dets_low)
        # unmatched low score detections will be discarded as background
        matches, u_tracks, u_dets_low = matching.linear_assignment(dists, thresh=0.5)  # TODO: Why 0.5 here?

        # Update tracks matched successfully
        for itrack, idet in matches:
            track = r_tracked_stracks[itrack]
            if track.state == TrackState.Tracked:  # always True
                track.update(dets_low[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(dets_low[idet], self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Put activated tracks yet unmatched so far into the list of lost tracks
        for itrack in u_tracks:
            track = r_tracked_stracks[itrack]
            track.mark_lost()
            lost_stracks.append(track)

        """ Step 3: Deal with newly initialized (unactivated) tracks with only one beginning frame"""
        # Unmatched high score detection after the first association
        dets_high = [dets_high[i] for i in u_dets_high]
        # Newly initialized tracks from high score detections in the last frame
        unconfirmed_tracks = [
            track for track in self.tracked_stracks if not track.is_activated
        ]
        # Matching
        dists = matching.iou_distance(unconfirmed_tracks, dets_high)
        if not self.args.mot20:  # TODO: Why fusing here again?
            dists = matching.fuse_score(dists, dets_high)
        matches, u_tracks, u_dets_high = matching.linear_assignment(dists, thresh=0.7)  # TODO: Why 0.7 here?

        # Step 3.1: Activate a new track if it associates with a high score detection unmatched after the first association
        for itrack, idet in matches:
            track = unconfirmed_tracks[itrack]
            track.update(dets_high[idet], self.frame_id)
            activated_starcks.append(track)

        # Step 3.2: Remove an unmatched track initialized in the last frame
        for itrack in u_tracks:
            track = unconfirmed_tracks[itrack]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 3.3: Initialize new tracks from unmatched high score detections in the current frame
        for idet in u_dets_high:
            track = dets_high[idet]
            # TODO: In the official repo, the threshold below is added by 0.1 (the reason is unknown)
            if track.score >= self.args.track_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

        """ Step 4: Remove tracks which have no matches over `self.max_time_lost` consecutive frames"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Step 5: Update tracks state"""
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # TODO: Why need to remove duplicate tracks between tracked and lost tracks? (taken from FairMOT)
        if self.args.with_deduplication:
            self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
