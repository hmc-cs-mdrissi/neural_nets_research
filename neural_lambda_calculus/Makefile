interpreter: LambdaMain.hs LambdaParser.hs LambdaInterp.hs
	ghc LambdaMain.hs -main-is LambdaMain.main -o interp

clean:
	rm -f *.o *.hi interp

generateArbitrary: GenerateArbitrarySimplyTypedLambda.hs ArbitrarySimplyTypedLambda.hs SimplyTypedLambdaParser.hs
	ghc GenerateArbitrarySimplyTypedLambda.hs -main-is GenerateArbitrarySimplyTypedLambda.main -o generateArbitrary

cleanGen:
	rm -f *.o *.hi generateArbitrary

cleanjson:
	rm -f *.json