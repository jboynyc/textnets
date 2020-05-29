import pandas as pd


moon_landing = pd.DataFrame(
    {'paper': ['The Guardian',
               'New York Times',
               'Boston Globe',
               'Houston Chronicle',
               'Washington Post',
               'Chicago Tribune',
               'Los Angeles Times'],
     'headline': ['3:56 am: Man Steps On to the Moon',
                  'Men Walk on Moon -- Astronauts Land on Plain, Collect Rocks, Plant Flag',
                  'Man Walks on Moon',
                  'Armstrong and Aldrich "Take One Small Step for Man" on the Moon',
                  'The Eagle Has Landed -- Two Men Walk on the Moon',
                  'Giant Leap for Mankind -- Armstrong Takes 1st Step on Moon',
                  'Walk on Moon -- That\'s One Small Step for Man, One Giant Leap for Mankind']
    }).set_index('paper')
