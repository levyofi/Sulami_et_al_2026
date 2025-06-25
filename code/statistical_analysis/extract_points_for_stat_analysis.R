library(terra)
library(dplyr)
library(sf)
library(rlist)


# your features and base directory
features         <- c('height','shade','real_solar','skyview','TGI', 'slope')
feature_maps_dir <- '~/Dropbox/idan/complete_subimages_cropped'

# which maps to process
df_test_maps <- read.csv('~/Dropbox/pycharm_projects/Sulami_et_al_Ecology/data/desert_maps.csv') %>% 
  filter(validation == 1)

# define a 31×31 moving window
w31 <- matrix(1, nrow = 31, ncol = 31)

# prepare a list to collect each map’s samples
results_list <- vector("list", length = nrow(df_test_maps))
names(results_list) <- df_test_maps$Map

library(terra)
library(dplyr)

# now include "slope" in your features
features         <- c('height','shade','real_solar','skyview','TGI','slope')
feature_maps_dir <- '~/Dropbox/idan/complete_subimages_cropped'
error_maps_dir <- '~/Dropbox/pycharm_projects/Sulami_et_al_Ecology/predicted_maps'
df_test_maps     <- read.csv('~/Dropbox/pycharm_projects/Sulami_et_al_Ecology/data/desert_maps.csv') %>%
  filter(test == 1)

all_maps_out <- list()

sizes = c(5, 9, 15, 21, 31, 47, 63, 81)

for (size in sizes){

  for(map_name in df_test_maps$Map){
    
    for (i in 1:5){
      # 1) sample 20 cells once per map
      print(paste("Size:",size, "Map:", map_name, i))
      model_errors <-rast(file.path(error_maps_dir, map_name, paste0("error_size_",size, "_", i, ".tif")))
      samp <- spatSample(model_errors, size=20, method="regular", xy=TRUE, as.df=T, exact = T)
      errors = samp[,paste0("error_size_",size, "_", i)]
      df_pts <- samp %>% 
        mutate(Map  = map_name, size = size,
               errors = errors) %>%
        select(!starts_with("error_size_"))
      
      # get point circles for the slope sd calculation
      pts_vect <- vect(data.frame(x = samp$x, y = samp$y),
                       geom = c("x","y"),
                       crs  = crs(model_errors))
      # 2. make a 31×31‐pixel buffer around each point
      px      <- res(model_errors)[1]        # pixel width
      radius  <- 2#(31/2) * px            # half‐width in map units
      circles <- terra::buffer(pts_vect, radius)
      
      
      for(feat in features){
        
        #get the data for each sampled coordinate
        if (feat != "slope") {
          r_feat <- rast(file.path(feature_maps_dir, map_name,
                                   paste0(feat, "_", i, ".tif")))
          vals <- terra::extract(r_feat, pts_vect, na.rm = TRUE, exact = T)
        } else { #need the sd of around each coordinate
          r_slope <- rast(file.path(feature_maps_dir, map_name,
                                    paste0("slope_", i, ".tif")))
          vals = terra::extract(r_slope, circles, fun = sd, na.rm = TRUE)
        }
        
        #create the table and add to the list
        df_data = data.frame(
          df_pts,
          feature    = feat,
          map_number = i,
          value      = vals[,2]
        )
        all_maps_out = list.append(all_maps_out, df_data)
      }
    }
  } 
}

# combine all maps into one big table
save(all_maps_out, file = "20_pixels_per_map_data.RData")
df_all_samples <- bind_rows(all_maps_out)

write.csv(df_all_samples, file = "20_pixels_per_map_data.csv", row.names = F)
