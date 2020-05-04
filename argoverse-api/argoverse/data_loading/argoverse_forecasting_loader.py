# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["ArgoverseForecastingLoader"]


@lru_cache(128)
def _read_csv(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """A caching CSV reader

    Args:
        path: Path to the csv file
        *args, **kwargs: optional arguments to be used while data loading

    Returns:
        pandas DataFrame containing the loaded csv
    """
    # print(path)
    return pd.read_csv(path, *args, **kwargs)


class ArgoverseForecastingLoader:
    def __init__(self, root_dir: Union[str, Path]):
        """Initialization function for the class.
 
        Args:
            root_dir: Path to the folder having sequence csv files
        """
        self._track_id_list: Optional[Mapping[str, Sequence[int]]] = None
        self._city_list: Optional[Mapping[str, str]] = None

        self.counter: int = 0

        root_dir = Path(root_dir)
        self.seq_list: Sequence[Path] = [(root_dir / x).absolute() for x in os.listdir(root_dir)]

        self.current_seq: Path = self.seq_list[self.counter]

    @property
    def track_id_list(self) -> Sequence[int]:
        """Get the track ids in the current sequence.

        Returns:
            list of track ids in the current sequence
        """
        if self._track_id_list is None:
            self._track_id_list = {}
            for seq in self.seq_list:
                seq_df = _read_csv(seq)
                self._track_id_list[str(seq)] = np.unique(seq_df["TRACK_ID"].values).tolist()

        return self._track_id_list[str(self.current_seq)]

    @property
    def city(self) -> str:
        """Get the city name for the current sequence.

        Returns:
            city name, i.e., either 'PIT' or 'MIA'
        """
        # if self._city_list is None:
        #     self._city_list = {}
        #     for seq in self.seq_list:
        #         seq_df = _read_csv(seq)
        #         self._city_list[str(seq)] = seq_df["CITY_NAME"].values[0]

        # return self._city_list[str(self.current_seq)]
        return self.seq_df["CITY_NAME"].values[0]

    @property
    def num_tracks(self) -> int:
        """Get the number of tracks in the current sequence. 

        Returns:
            number of tracks in the current sequence
        """
        return len(self.track_id_list)

    @property
    def seq_df(self) -> pd.DataFrame:
        """Get the dataframe for the current sequence.

        Returns:
            pandas DataFrame for the current sequence
        """
        return _read_csv(self.current_seq)

    @property
    def agent_traj(self) -> np.ndarray:
        """Get the trajectory for the track of type 'AGENT' in the current sequence.

        Returns:
            numpy array of shape (seq_len x 2) for the agent trajectory
        """
        agent_x = self.seq_df[self.seq_df["OBJECT_TYPE"] == "AGENT"]["X"]
        agent_y = self.seq_df[self.seq_df["OBJECT_TYPE"] == "AGENT"]["Y"]
        agent_traj = np.column_stack((agent_x, agent_y))
        return agent_traj

    def traj_with_track_id(self,track_id):
        """return trajectory of a particular traj id"""
        # import pdb; pdb.set_trace()
        time_stamp_agent=self.seq_df[self.seq_df["OBJECT_TYPE"] == "AGENT"]["TIMESTAMP"].tolist()
        
        index_time_stamp=list(range(50))
        
        time_to_index=dict(zip(time_stamp_agent,index_time_stamp))
        
        time_stamp_traj=self.seq_df[self.seq_df["TRACK_ID"] == track_id]["TIMESTAMP"].tolist()
        index_traj=[time_to_index[x] for x in time_stamp_traj]


#         print("index_traj", index_traj)
#         print("$ 1 $: ", [i>20 for i in index_traj])
#         print("$ 2 $: ",sum([i>20 for i in index_traj]))
        
#         if 15 in index_traj and sum([i>20 for i in index_traj])>5:
        if sum([i>-1 for i in index_traj])==50:
#             print("$ 3 $: ", sum([i>20 for i in index_traj])>5)
#             print(sum([i>20 for i in index_traj])>5)
            
            agent_x = self.seq_df[self.seq_df["TRACK_ID"] == track_id]["X"]
            agent_y = self.seq_df[self.seq_df["TRACK_ID"] == track_id]["Y"]

            # agent_traj=np.array([[np.nan]*2]*50)
            # import pdb; pdb.set_trace()
            agent_x=np.interp(list(range(50)),index_traj,agent_x)
            agent_y=np.interp(list(range(50)),index_traj,agent_y)
            agent_traj = np.column_stack((agent_x, agent_y))
            
            # print(f"The shape of neighbour trajectory for {track_id} is", agent_traj.shape)
            # print(f"Start index: {start_index}. End index: {end_index}")
            # if (len(agent_x) != end_index-start_index+1):
            #     import pdb; pdb.set_trace()

            #     assert (len(agent_x)==end_index-start_index+1),"Gap between the track indexes"
            # import pdb; pdb.set_trace()
            return agent_traj
        else:
            return None
        
    def track_id_list_neighbours(self) -> Sequence[int]:
        """Get the track ids in the current sequence.

        Returns:
            list of track ids in the current sequence
        """
        # if self._track_id_list is None:
        #     self._track_id_list = {}
        #     for seq in self.seq_list:
        #         seq_df = _read_csv(seq)
        #         self._track_id_list[str(seq)] = np.unique(seq_df["TRACK_ID"].values).tolist()
        # import pdb; pdb.set_trace()
        return np.unique(self.seq_df[self.seq_df["OBJECT_TYPE"]=="OTHERS"]["TRACK_ID"].tolist())

    def neighbour_traj(self):
        track_ids=self.track_id_list_neighbours()
        neighbours_traj=[]
        for track_id in track_ids:
            traj=self.traj_with_track_id(track_id)
            # print(traj)
            if traj is not None:
                neighbours_traj.append(traj)
#         print("neighbours_traj:", neighbours_traj)        
        return neighbours_traj
        # if len(neighbours_traj)<=1:
        #     return neighbours_traj
        
        """
        Experimenting with just one closest neighbour
        """
        # closest_neighbour=neighbours_traj[0]
        # # import pdb; pdb.set_trace()
        # # print(closest_neighbour.shape)
        # for curr_neighbour in neighbours_traj:
        #     norm_closest_neigh=np.linalg.norm([closest_neighbour[-1,0]-self.agent_traj[19,0],closest_neighbour[-1,1]-self.agent_traj[19,1]])
        #     norm_curr_neigh=np.linalg.norm([curr_neighbour[-1,0]-self.agent_traj[19,0],curr_neighbour[-1,1]-self.agent_traj[19,1]])
        #     if norm_closest_neigh>norm_curr_neigh:
        #         closest_neighbour=curr_neighbour
        #         # print("Closest distance of neighbour is ", norm_closest_neigh)
        # # print(closest_neighbour.shape)
        # # print("Final Closest distance of neighbour is ", norm_closest_neigh)
        # return [closest_neighbour]
        # # import pdb; pdb.set_trace()
        # # print("Size of neighbours trajectiry is",len(neighbours_traj))
        # # return neighbours_traj
    def __iter__(self) -> "ArgoverseForecastingLoader":
        """Iterator for enumerating over sequences in the root_dir specified.

        Returns:
            Data Loader object for the first sequence in the data
        """
        self.counter = 0
        return self

    def __next__(self) -> "ArgoverseForecastingLoader":
        """Get the Data Loader object for the next sequence in the data.

        Returns:
            Data Loader object for the next sequence in the data
        """
        if self.counter >= len(self):
            raise StopIteration
        else:
            self.current_seq = self.seq_list[self.counter]
            self.counter += 1
            return self

    def __len__(self) -> int:
        """Get the number of sequences in the data

        Returns:
            Number of sequences in the data
        """
        return len(self.seq_list)

    def __str__(self) -> str:
        """Decorator that returns a string storing some stats of the current sequence

        Returns:
            A string storing some stats of the current sequence
        """
        return f"""Seq : {self.current_seq}
        ----------------------
        || City: {self.city}
        || # Tracks: {len(self.track_id_list)}
        ----------------------"""

    def __getitem__(self, key: int) -> "ArgoverseForecastingLoader":
        """Get the DataLoader object for the sequence corresponding to the given index.

        Args:
            key: index of the element

        Returns:
            Data Loader object for the given index
        """

        self.counter = key
        self.current_seq = self.seq_list[self.counter]
        return self

    def get(self, seq_id: Union[Path, str]) -> "ArgoverseForecastingLoader":
        """Get the DataLoader object for the given sequence path.

        Args:
            seq_id: Fully qualified path to the sequence

        Returns:
            Data Loader object for the given sequence path
        """
        self.current_seq = Path(seq_id).absolute()
        return self
