parseCodebase: ParseRealCodebases.hs
	ghc ParseRealCodebases.hs -main-is ParseRealCodebases.main -o ParseRealCodebases;
	./ParseRealCodebases $(input_file) > $(output_file)

cleanCSharpData: CleanCSharpData.hs
	ghc CleanCSharpData.hs -main-is CleanCSharpData.main -o CleanCSharpData;
	./CleanCSharpData $(input_file) > $(output_file)

clean:
	rm *.hi *.o ParseRealCodebases CleanCSharpData