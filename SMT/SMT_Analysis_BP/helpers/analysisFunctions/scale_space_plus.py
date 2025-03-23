"""
Suite of functions and classes to perfrom the scale space plus procedure to create the reconstruction image for scale space analysis.

"""

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import skimage as skimage
from scipy.stats import multivariate_normal

from SMT.SMT_Analysis_BP.helpers.analysisFunctions.Analysis_functions import (
    rescale_range,
)

KEY_IMAGE = {
    "png": skimage.io.imsave,
    "jpg": skimage.io.imsave,
    "tif": skimage.io.imsave,
    "svg": skimage.io.imsave,
}
MASK_VALUE = 255
BOUNDING_BOX_PADDING = 5
CONVERSION_TYPES = {"RC_to_Original": 0, "original_to_RC": 1}
RANDOM_SEED = 666  # for reproducibility, also praise the devil (joking)


# numpy version of get_gaussian
def get_gaussian(mu, sigma, domain=[list(range(10)), list(range(10))]):
    """
    Parameters
    ----------
    mu : array-like or float of floats
        center position of gaussian (x,y) or collection of (x,y)
    sigma : float or array-like of floats of shape mu
        sigma of the gaussian
    domain : array-like, Defaults to 0->9 for x,y
        x,y domain over which this gassuain is over


    Returns
    -------
    array-like 2D
        values of the gaussian centered at mu with sigma across the (x,y) points defined in domain

    Notes:
    ------
    THIS IS IMPORTANT: MAKE SURE THE TYPES IN EACH PARAMETER ARE THE SAME!!!!
    """
    # generate a multivariate normal distribution with the given mu and sigma over the domain using scipy stats
    # generate the grid
    x = domain[0]
    y = domain[1]
    xx, yy = np.meshgrid(x, y)
    # generate the multivariate normal distribution
    rv = multivariate_normal(mu, sigma)
    # generate the probability distribution
    gauss = rv.pdf(np.dstack((xx, yy)))
    # reshape the distribution on the grid
    return gauss


# define a ABC class for the scale space plus procedure
class ScaleSpacePlus(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def make_reconstruction(self):
        NotImplementedError(
            "The make_reconstruction method must be implemented in the child class"
        )

    @abstractmethod
    def saving_image(self):
        NotImplementedError(
            "The saving_image method must be implemented in the child class"
        )

    @abstractmethod
    def print_state(self):
        NotImplementedError(
            "The print_state method must be implemented in the child class"
        )


class SM_reconstruction_image(ScaleSpacePlus):
    def __init__(
        self,
        img_dims_normal: tuple | list,
        pixel_size_normal: int | float = 130,
        rescale_pixel_size: int = 10,
    ) -> None:
        self.img_dims_normal = img_dims_normal  # in pixels
        self.pixel_size_normal = pixel_size_normal  # in nm
        self.rescale_pixel_size = rescale_pixel_size  # in nm
        self.img_dims = [
            int(i * self.pixel_size_normal / self.rescale_pixel_size)
            for i in self.img_dims_normal
        ]
        self.domain = [
            np.arange(0, self.img_dims[0], 1.0),
            np.arange(0, self.img_dims[1], 1.0),
        ]

    def print_state(self):
        pass

    def make_reconstruction(
        self, localizations: np.ndarray, localization_error: np.ndarray | float
    ):
        """
        Parameters:
        -----------
        localizations: np.ndarray
            Array of localizations in the form of (x,y) in pixels (original scale)
        localization_error: np.ndarray or float
            Array of localization error in nm or a scalar

        Returns:
        --------
        img_space: np.ndarray
            Image space of the reconstruction
        """
        # create a pandas dataframe for the localizations and localization error
        # check if the localization error is a scalar or an array of the same length as the localizations
        if np.isscalar(localization_error):
            localization_error = np.ones(len(localizations)) * localization_error
        elif len(localization_error) != len(localizations):
            raise ValueError(
                "The length of the localization error must be the same as the length of the localizations"
            )
        self.df_localizations = pd.DataFrame(
            {
                "x": localizations[:, 0],
                "y": localizations[:, 1],
                "localization_error": localization_error,
            }
        )
        # reformat to be in the form of a collection of (x,y)
        self.df_localizations = self.df_localizations[["x", "y", "localization_error"]]
        # rescale the localizations
        self.df_localizations[["x", "y"]] = (
            self.df_localizations[["x", "y"]]
            * self.pixel_size_normal
            / self.rescale_pixel_size
        )
        # loop over the localizations
        self.img_space = np.zeros(self.img_dims)

        for i in range(len(self.df_localizations)):
            x = self.df_localizations.iloc[i]["x"]
            y = self.df_localizations.iloc[i]["y"]
            loc_error_val = self.df_localizations.iloc[i]["localization_error"] / (
                self.rescale_pixel_size
            )
            sigma_shape = np.ones(2) * loc_error_val

            # we want to only sample a 10x10 pixel area around the localization and then remap it to the original image space
            # make the new domain range scale with this localization error/ pixel size ratio
            domain_max = int(5 * loc_error_val)
            domain = [np.arange(0, domain_max, 1.0), np.arange(0, domain_max, 1.0)]
            # lets get the x,y in this new domain
            x_scaled = rescale_range(x, *[0, self.img_dims[0]], *[0, domain_max])
            y_scaled = rescale_range(y, *[0, self.img_dims[1]], *[0, domain_max])
            # get the gaussian
            temp_scape = get_gaussian(
                mu=[x_scaled, y_scaled], sigma=sigma_shape, domain=domain
            )
            # temp_scape = get_gaussian(
            #     mu=[domain_max/2.0, domain_max/2.0], sigma=sigma_shape, domain=domain
            # )
            # now we have a 20x20 matrix but we need to embed it into the orginal self.img_space
            # we can find the bottom corner of the 20x20 matrix in the self.img_space
            x_bottom = int(x - domain_max / 2.0)
            y_bottom = int(y - domain_max / 2.0)
            # now we can embed the 20x20 matrix into the self.img_space
            try:
                self.img_space[
                    x_bottom : x_bottom + domain_max, y_bottom : y_bottom + domain_max
                ] = self.img_space[
                    x_bottom : x_bottom + domain_max, y_bottom : y_bottom + domain_max
                ] + np.array(temp_scape)
            except:
                print(
                    "The localization is too close to the edge of the image space. Edge cases are not implimented yet so the localization is not included in the reconstruction."
                )
        return self.img_space.T

    def saving_image(self, full_path: str, name, type: str):
        """
        Parameters:
        -----------
        full_path: str
            Full path to save the image
        name: str
            Name of the image
        type: str
            Type of the image. Supported:
                - png
                - jpg
                - tif
                - svg
        """
        # check if the type is supported
        if type not in KEY_IMAGE.keys():
            raise ValueError("The type of the image is not supported")
        # check if the path exists
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        # make the name by joining the strings
        name = os.path.join(full_path, name + "." + type)
        # save the image
        skimage.io.imsave(name, self.img_space.T)

    @property
    def img_dims(self):
        return self._img_dims

    @img_dims.setter
    def img_dims(self, img_dims):
        self._img_dims = img_dims

    @property
    def img_space(self):
        if not hasattr(self, "_img_space"):
            self._img_space = np.zeros(self.img_dims)
        return self._img_space

    @img_space.setter
    def img_space(self, img_space):
        self._img_space = img_space

    @property
    def domain(self):
        if not hasattr(self, "_domain"):
            self._domain = [
                np.arange(0, self.img_dims[0], 1.0),
                np.arange(0, self.img_dims[1], 1.0),
            ]
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain

    @property
    def df_localizations(self):
        return self._df_localizations

    @df_localizations.setter
    def df_localizations(self, df_localizations):
        self._df_localizations = df_localizations

    @property
    def total_localizations(self):
        return len(self.df_localizations)


class SM_reconstruction_masked(ScaleSpacePlus):
    def __init__(
        self,
        img_dims_normal: tuple | list,
        pixel_size_normal: int | float = 130,
        rescale_pixel_size: int = 10,
    ) -> None:
        self.img_dims_normal = img_dims_normal  # in pixels
        self.pixel_size_normal = pixel_size_normal  # in nm
        self.rescale_pixel_size = rescale_pixel_size  # in nm
        self._img_dims = [
            int(i * self.pixel_size_normal / self.rescale_pixel_size)
            for i in self.img_dims_normal
        ]
        self.normal_domain = [
            np.arange(0, self.img_dims[0], 1.0),
            np.arange(0, self.img_dims[1], 1.0),
        ]

    def print_state(self):
        pass

    def make_reconstruction(self, localizations, localization_error, masked_img):
        # set up the masked domain
        self._setup_masked_domain(mask_value=MASK_VALUE, masked_img_space=masked_img)
        self._setup_bounding_box()

        # create a pandas dataframe for the localizations and localization error
        # check if the localization error is a scalar or an array of the same length as the localizations
        if np.isscalar(localization_error):
            localization_error = np.ones(len(localizations)) * localization_error
        elif len(localization_error) != len(localizations):
            raise ValueError(
                "The length of the localization error must be the same as the length of the localizations"
            )
        self.df_localizations = pd.DataFrame(
            {
                "x": localizations[:, 0],
                "y": localizations[:, 1],
                "localization_error": localization_error,
            }
        )
        # reformat to be in the form of a collection of (x,y)
        df_localizations = self.df_localizations[["x", "y", "localization_error"]]
        # since we want to make a smaller image in the form of the masked domain we need to rescale the localizations to the masked domain
        # just subtract the min of the masked domain from the localizations
        df_localizations[["x", "y"]] = (
            self.df_localizations[["x", "y"]]
            - np.min(self.masked_domain, axis=0)
            + BOUNDING_BOX_PADDING
        )

        # rescale the localizations
        df_localizations[["x", "y"]] = (
            df_localizations[["x", "y"]]
            * self.pixel_size_normal
            / self.rescale_pixel_size
        )

        # now make an empty image space of the masked domain using the bounding box
        small_img_space = np.zeros(
            [
                int(
                    (self.bounding_box[1, 0] - self.bounding_box[0, 0])
                    * self.pixel_size_normal
                    / self.rescale_pixel_size
                ),
                int(
                    (self.bounding_box[1, 1] - self.bounding_box[0, 1])
                    * self.pixel_size_normal
                    / self.rescale_pixel_size
                ),
            ]
        )
        domain_small = [
            np.arange(0, small_img_space.shape[0], 1.0),
            np.arange(0, small_img_space.shape[1], 1.0),
        ]

        # loop over the localizations
        for i in range(len(df_localizations)):
            x = df_localizations.iloc[i]["x"]
            y = df_localizations.iloc[i]["y"]
            loc_error_val = df_localizations.iloc[i]["localization_error"] / (
                self.rescale_pixel_size
            )
            sigma_shape = np.ones(2) * loc_error_val

            # we want to only sample a 10x10 pixel area around the localization and then remap it to the original image space
            # make the new domain range scale with this localization error/ pixel size ratio
            domain_max = int(
                5 * loc_error_val
            )  # this is an issue maybe? Int cuts off part of the gaussian asymetrically.
            domain = [np.arange(0, domain_max, 1.0), np.arange(0, domain_max, 1.0)]
            # lets get the x,y in this new domain
            x_scaled = rescale_range(
                x, *[0, small_img_space.shape[0]], *[0, domain_max]
            )
            y_scaled = rescale_range(
                y, *[0, small_img_space.shape[1]], *[0, domain_max]
            )
            # get the gaussian
            temp_scape = get_gaussian(
                mu=[x_scaled, y_scaled], sigma=sigma_shape, domain=domain
            )
            # temp_scape = get_gaussian(
            #     mu=[domain_max/2.0, domain_max/2.0], sigma=sigma_shape, domain=domain
            # )
            # now we have a 20x20 matrix but we need to embed it into the orginal self.img_space
            # we can find the bottom corner of the 20x20 matrix in the self.img_space
            x_bottom = int(x - domain_max / 2.0)
            y_bottom = int(y - domain_max / 2.0)
            # now we can embed the 20x20 matrix into the self.img_space
            try:
                small_img_space[
                    x_bottom : x_bottom + domain_max, y_bottom : y_bottom + domain_max
                ] = small_img_space[
                    x_bottom : x_bottom + domain_max, y_bottom : y_bottom + domain_max
                ] + np.array(temp_scape)
            except:
                print(
                    "The localization is too close to the edge of the image space. Edge cases are not implimented yet so the localization is not included in the reconstruction."
                )

        return small_img_space.T

    def make_uniform_reconstruction(
        self, localizations, localization_error, masked_img
    ):
        """
        We will now use the number of localizations to uniformly make a reconstruction image on the masked domain.
        """
        # len of the localizations
        num_localizations = len(localizations)
        if num_localizations == 0:
            print(
                "There are no localizations to make a reconstruction from, so using an empty image space"
            )
            # return an empty image space
            return np.zeros(
                [
                    int(
                        (self.bounding_box[1, 0] - self.bounding_box[0, 0])
                        * self.pixel_size_normal
                        / self.rescale_pixel_size
                    ),
                    int(
                        (self.bounding_box[1, 1] - self.bounding_box[0, 1])
                        * self.pixel_size_normal
                        / self.rescale_pixel_size
                    ),
                ]
            ).T
        # correct the localization error
        if np.isscalar(localization_error):
            localization_error = np.ones(num_localizations) * localization_error
        elif len(localization_error) != num_localizations:
            raise ValueError(
                "The length of the localization error must be the same as the length of the localizations"
            )

        # set up the masked domain
        self._setup_masked_domain(mask_value=MASK_VALUE, masked_img_space=masked_img)
        self._setup_bounding_box()

        # now make an empty image space of the masked domain using the bounding box
        small_img_space = np.zeros(
            [
                int(
                    (self.bounding_box[1, 0] - self.bounding_box[0, 0])
                    * self.pixel_size_normal
                    / self.rescale_pixel_size
                ),
                int(
                    (self.bounding_box[1, 1] - self.bounding_box[0, 1])
                    * self.pixel_size_normal
                    / self.rescale_pixel_size
                ),
            ]
        )
        domain_small = [
            np.arange(0, small_img_space.shape[0], 1.0),
            np.arange(0, small_img_space.shape[1], 1.0),
        ]
        # choose a random localization based on the masked domain
        df_localizations = self._get_uniform_localization(
            masked_domain=self.masked_domain, num_localizations=num_localizations
        )
        # move the localizations based on the domain of the bounding box
        df_localizations[["x", "y"]] = (
            df_localizations[["x", "y"]]
            - np.min(self.masked_domain, axis=0)
            + BOUNDING_BOX_PADDING
        )
        # rescale the localizations
        df_localizations[["x", "y"]] = (
            df_localizations[["x", "y"]]
            * self.pixel_size_normal
            / self.rescale_pixel_size
        )
        # now we can make the reconstruction
        # loop over the localizations
        for i in range(len(df_localizations)):
            x = df_localizations.iloc[i]["x"]
            y = df_localizations.iloc[i]["y"]
            loc_error_val = localization_error[i] / (self.rescale_pixel_size)
            sigma_shape = np.ones(2) * loc_error_val

            # we want to only sample a 10x10 pixel area around the localization and then remap it to the original image space
            # make the new domain range scale with this localization error/ pixel size ratio
            domain_max = int(5 * loc_error_val)
            domain = [np.arange(0, domain_max, 1.0), np.arange(0, domain_max, 1.0)]
            # lets get the x,y in this new domain
            x_scaled = rescale_range(
                x, *[0, small_img_space.shape[0]], *[0, domain_max]
            )
            y_scaled = rescale_range(
                y, *[0, small_img_space.shape[1]], *[0, domain_max]
            )
            # get the gaussian
            temp_scape = get_gaussian(
                mu=[x_scaled, y_scaled], sigma=sigma_shape, domain=domain
            )
            # temp_scape = get_gaussian(
            #     mu=[domain_max/2.0, domain_max/2.0], sigma=sigma_shape, domain=domain
            # )
            # now we have a 20x20 matrix but we need to embed it into the orginal self.img_space
            # we can find the bottom corner of the 20x20 matrix in the self.img_space
            x_bottom = int(x - domain_max / 2.0)
            y_bottom = int(y - domain_max / 2.0)
            # now we can embed the 20x20 matrix into the self.img_space
            try:
                small_img_space[
                    x_bottom : x_bottom + domain_max, y_bottom : y_bottom + domain_max
                ] = small_img_space[
                    x_bottom : x_bottom + domain_max, y_bottom : y_bottom + domain_max
                ] + np.array(temp_scape)
            except:
                print(
                    "The localization is too close to the edge of the image space. Edge cases are not implimented yet so the localization is not included in the reconstruction."
                )

        return small_img_space.T

    def _get_uniform_localization(self, masked_domain, num_localizations):
        # first we need to make a choice on the x,y pixel location
        # we can do this by choosing a random index from the masked domain
        # use the random seed for reproducibility, legacy implimentation from numpy
        np.random.seed(RANDOM_SEED)
        choice_indx = np.random.choice(
            np.arange(0, len(masked_domain), 1), size=num_localizations, replace=True
        )
        # now we can get the x,y pixel location for each localization
        x = masked_domain[choice_indx, 0]
        y = masked_domain[choice_indx, 1]
        # since these are integers we need to add a random number between 0 and 1 to each x,y
        x = x + np.random.rand(len(x))
        y = y + np.random.rand(len(y))
        # now we can make the dataframe
        df_localizations = pd.DataFrame({"x": x, "y": y})
        return df_localizations

    def saving_image(self, img: np.ndarray, full_path: str, type: str = None):
        """
        Save the given image to the full path with the given name and type.

        Parameters:
        -----------
        img: np.ndarray
            The image to save
        full_path: str
            Full path to save the image (this includes the name)
        type: str
            Type of the image. Supported:
                - png
                - jpg
                - tif
                - svg
        """
        # the path should exist so don't check
        # save the image
        skimage.io.imsave(full_path, img)

    def coordinate_conversion(
        self, spatial_dim: np.ndarray, radius: np.ndarray, conversion_type: str
    ):
        """
        Converts the spatial dimension of the reconstruction image to the original image space.
        Parameters:
        -----------
        spatial_dim: ndarray
            The spatial dimension to convert. This is the dimension of the reconstruction image space.
        radius: ndarray
            The radius of the reconstruction image space.
        conversion_type: str
            The type of conversion to perform. Supported:
                - RC_to_Original: converts the reconstruction image space to the original image space
                - original_to_RC: converts the original image space to the reconstruction image space

        Returns:
        --------
        converted_dim: ndarray
            The converted spatial dimension
        convert_radius: ndarray
            The converted radius
        """
        # check if the conversion type is supported
        if conversion_type not in CONVERSION_TYPES.keys():
            raise ValueError("The conversion type is not supported")
        # convert the spatial dimension
        if conversion_type == "RC_to_Original":
            # take into account the padding and the masked domain and the rescale pixel size
            # divide by the rescale pixel size
            converted_dim = (
                spatial_dim * self.rescale_pixel_size / self.pixel_size_normal
            )
            # add the padding
            converted_dim = converted_dim - BOUNDING_BOX_PADDING
            # add the masked domain
            converted_dim = converted_dim + np.min(self.masked_domain, axis=0)
            convert_radius = radius * self.rescale_pixel_size / self.pixel_size_normal
        elif conversion_type == "original_to_RC":
            # take into account the padding and the masked domain and the rescale pixel size
            # subtract the masked domain
            converted_dim = spatial_dim - np.min(self.masked_domain, axis=0)
            # add the padding
            converted_dim = converted_dim + BOUNDING_BOX_PADDING
            # multiply by the rescale pixel size
            converted_dim = (
                converted_dim * self.pixel_size_normal / self.rescale_pixel_size
            )
            convert_radius = radius * self.pixel_size_normal / self.rescale_pixel_size
        return converted_dim, convert_radius

    def _setup_masked_domain(
        self, mask_value: int, masked_img_space: np.ndarray
    ) -> None:
        """
        Setups the masked domain for the reconstruction. This is the domain of the image space which represents a cell area in which the localizations are located.
        Parameters:
        -----------
        mask_value: int
            The value in the masked_img_space which represents the cell area (usually 1)
        masked_img_space: np.ndarray
            The masked image space. If None then the attribute masked_img_space is used. If the attribute does not exist then an error is raised.
            This is a 2D array defining the frame of view of the whole img

        Sets:
        -----
        self._masked_domain: np.ndarray
            The domain of the masked image space which represents the cell area in which the localizations are located.
            The format is:
            [[x,y],[x,y],...,[x,y]] where x,y represent the index and hence the pixel location in the masked image space which represents the cell area.
        """
        if masked_img_space is not None:
            self._masked_img_space = masked_img_space
        # now we can create the masked domain
        # we want to find the [[x,y],[x,y]] coordinates of the masked_img_space which are a certain value defined by mask_value
        indxes = np.where(self.masked_img_space == mask_value)
        domain_xy = np.array(
            [indxes[1], indxes[0]]
        ).T  # this is to correct for the fact that img space and where invert the x,y axis
        # now we can create the masked domain
        self._masked_domain = domain_xy

    def _setup_bounding_box(self) -> None:
        """
        Sets up the bounding box for the reconstruction. This is the bounding box of the masked domain.
        Sets:
        -----
        self._bounding_box: np.ndarray
            The bounding box of the masked domain. The format is:
            [[x_min,y_min],[x_max,y_max]]
        """
        # check if the masked domain exists
        if not hasattr(self, "_masked_domain"):
            raise ValueError(
                "The masked_domain attribute does not exist. Please create it first. Or pass a masked_img_space to the _setup_masked_domain method"
            )
        # now we can create the bounding box
        self._bounding_box = np.array(
            [
                [
                    np.min(self.masked_domain[:, 0]) - BOUNDING_BOX_PADDING,
                    np.min(self.masked_domain[:, 1]) - BOUNDING_BOX_PADDING,
                ],
                [
                    np.max(self.masked_domain[:, 0]) + BOUNDING_BOX_PADDING,
                    np.max(self.masked_domain[:, 1]) + BOUNDING_BOX_PADDING,
                ],
            ]
        )

    @property
    def masked_img_space(self):
        return self._masked_img_space

    @property
    def img_dims(self):
        return self._img_dims

    @property
    def masked_domain(self):
        if not hasattr(self, "_masked_domain"):
            self._setup_masked_domain(mask_value=1)
        return self._masked_domain

    @property
    def bounding_box(self):
        if not hasattr(self, "_bounding_box"):
            self._setup_bounding_box()
        return self._bounding_box

    @property
    def df_localizations(self):
        return self._df_localizations

    @df_localizations.setter
    def df_localizations(self, df_localizations):
        self._df_localizations = df_localizations


############################################
# testing
############################################

# if __name__=="__main__":
#     cell_1_mask = "/Users/baljyot/Documents/SMT_Movies/testing_SM_recon/Movie_1/Cell_2/Mask_Cell_2.tif"
#     cell_1_loc = "/Users/baljyot/Documents/SMT_Movies/testing_SM_recon/Movie_1/Cell_2/randomMovie_1.tif_spots.csv"

#     #load the masked image
#     masked_img = skimage.io.imread(cell_1_mask)
#     #load the localizations
#     colnames = ['track_ID','x','y','frame','intensity']
#     df_localizations = pd.read_csv(cell_1_loc,usecols=(2,4,5,8,12),delimiter=',',skiprows=4,names=colnames)
#     #initialize the reconstruction class
#     recon = SM_reconstruction_masked(img_dims_normal=masked_img.shape,pixel_size_normal=130,rescale_pixel_size=130)
#     #make the reconstruction
#     recon_img = recon.make_reconstruction(localizations=df_localizations[['x','y']].values,localization_error=130,masked_img=masked_img)
#     #plot the reconstruction
#     plt.imshow(recon_img)
#     plt.show()
#     #make the uniform reconstruction
#     recon_img_uniform = recon.make_uniform_reconstruction(localizations=df_localizations[['x','y']].values,localization_error=130,masked_img=masked_img)
#     #plot the reconstruction
#     plt.imshow(recon_img_uniform)
#     plt.show()
#     #find the CM of the localizations in the original image space
#     cm = np.mean(df_localizations[['x','y']].values,axis=0)
#     #convert the CM to the reconstruction image space
#     cm_converted = recon.coordinate_conversion(spatial_dim=cm,radius=0,conversion_type='original_to_RC')
#     print(cm_converted,cm)
#     #save the image
#     #skimage.io.imsave("/Users/baljyot/Documents/SMT_Movies/testing_SM_recon/Movie_1/Cell_1/Reconstruction_Cell_1.tif",recon_img)
