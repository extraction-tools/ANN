{-# LANGUAGE ForeignFunctionInterface #-}

{-
TT
-}

module Neuron where

import Prelude
import Foreign.C.Types (CInt(..), CDouble(..))

-- Functions to get values of F, errF, and phi
foreign import ccall unsafe "getF" getF :: CInt -> CDouble
foreign import ccall unsafe "getErrF" getErrF :: CInt -> CDouble
foreign import ccall unsafe "getPhi" getPhi :: CInt -> CInt

getIndexedList :: (CInt -> a) -> [a]
getIndexedList f = map f [0 .. getNumIndexes - 1]

fList :: [CDouble]
fList = getIndexedList getF
errFList :: [CDouble]
errFList = getIndexedList getErrF
phiList :: [CInt]
phiList = getIndexedList getPhi


-- Functions to retrieve single values from the C file
foreign import ccall unsafe "getNumIndexes" getNumIndexes :: CInt
foreign import ccall unsafe "getNumReplicas" getNumReplicas :: CInt
foreign import ccall unsafe "getDVCS" getDVCS :: CDouble
foreign import ccall unsafe "getRand" getRand :: CInt -> CDouble

-- phi[index] -> BHUU
foreign import ccall unsafe "getBHUU" getBHUU :: CInt -> CDouble
-- phi[index] -> ReH -> ReE -> ReHtilde -> IUU
foreign import ccall unsafe "getIUU" getIUU :: CInt -> CDouble -> CDouble -> CDouble -> CDouble

-- reH_mean -> reH_stddev -> reE_mean -> reE_stddev -> reHtilde_mean -> reHtilde_stddev -> IO()
foreign import ccall unsafe "writeMeanAndStddev" writeMeanAndStddev :: CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> IO()


-- Box-Muller Transformation
-- Produces a number that follows a Gaussian distribution with mean=0 and stddev=1
boxMuller :: CInt -> CDouble
boxMuller a = sqrt (-2.0 * log (getRand a)) * sin (6.283185307179586 {- 2*pi -} * getRand (a + 1))

-- Calculate the mean percent error in F with the given values of ReH, ReE, ReHtilde, and the given replica
calcFError :: [CDouble] -> CDouble -> CDouble -> CDouble -> CDouble
calcFError replicas reH reE reHtilde = sum percentErrors / fromIntegral getNumIndexes
    where
        f_predicted = map (+ getDVCS) $ zipWith (+) (map getBHUU phiList) (map (\i -> getIUU i reH reE reHtilde) phiList)
        f_actual = zipWith (+) fList $ zipWith (*) errFList replicas
        percentErrors = map abs $ zipWith (/) (zipWith (-) f_actual f_predicted) f_actual

findMinInGrid :: [[CDouble]] -> [CDouble]
findMinInGrid x = aux (tail x) (head x)
    where
        aux :: [[CDouble]] -> [CDouble] -> [CDouble]
        aux xs minCFFs
            | null xs = minCFFs
            | head xs !! 3 < minCFFs !! 3 = aux (tail xs) (head xs)
            | otherwise = aux (tail xs) minCFFs

calcCFFs :: CInt -> [CDouble]
calcCFFs replicaNum = replicas --aux (totalDist / num) 1.0 1.0 1.0 9999.0
    where
        replicas = if replicaNum == -1 then [0.0 | _ <- [1 .. getNumIndexes]] else [boxMuller a | a <- [1 .. getNumIndexes]]
        num = 10 :: CDouble -- should actually be CInt
        totalDist = if replicaNum == -1 then 100.0 else 10.0 :: CDouble
        aux dist reHguess reEguess reHtildeguess bestError
            | dist >= 0.0001 = [reHguess, reEguess, reHtildeguess]
            | minInGrid !! 3 < bestError = aux dist (head minInGrid) (minInGrid !! 1) (minInGrid !! 2) (minInGrid !! 3)
            | otherwise = aux (dist / num) reHguess reEguess reHtildeguess bestError
                where
                    axes = map (* dist) [-num..num]
                    grid = [[reHguess + reHchange, reEguess + reEchange, reHtildeguess + reHtildechange, 
                             calcFError replicas (reHguess + reHchange) (reEguess + reEchange) (reHtildeguess + reHtildechange)] | 
                            reHchange <- axes, reEchange <- axes, reHtildechange <- axes]
                    minInGrid = findMinInGrid grid

calcMean :: [CDouble] -> CDouble
calcMean x = sum x / (realToFrac . length $ x)

calcStddev :: [CDouble] -> CDouble
calcStddev xs = sqrt $ sum (map (\i -> (i - average)**2) xs) / ((realToFrac . length $ xs) - 1)
    where
        average = calcMean xs

localFit :: IO()
localFit = 
    print arr
    --print cffs
    --writeMeanAndStddev (head (head cffs)) (head cffs !! 1) (head cffs !! 2) (head (cffs !! 1)) ((cffs !! 1) !! 1) ((cffs !! 1) !! 2)
    --writeMeanAndStddev (calcMean reH_list) (calcStddev reH_list) (calcMean reE_list) (calcStddev reE_list) (calcMean reHtilde_list) (calcStddev reHtilde_list)
    where
        cffs = map calcCFFs [0 .. getNumReplicas - 1]
        reH_list = [head x | x <- cffs]
        reE_list = [x !! 1 | x <- cffs]
        reHtilde_list = [x !! 2 | x <- cffs]
        arr = [boxMuller a | a <- [0..9999]]

foreign export ccall localFit :: IO()
