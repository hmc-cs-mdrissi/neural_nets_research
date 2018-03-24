{-# LANGUAGE DeriveGeneric #-}

module LambdaParser where

import Text.Parsec.Language (emptyDef)
import qualified Text.Parsec.Token as P
import Text.Parsec.Char
import Text.Parsec.Prim
import Text.Parsec.Combinator
import Text.Parsec.String (Parser)
import Text.Parsec.Error
import Control.Applicative (liftA2, some)
import Data.Aeson
import GHC.Generics
import ForLambdaCommon

data ProgLambda = UnitLambda | IfL Cmp ProgLambda ProgLambda | ExprL Expr
                | LetLambda String ProgLambda ProgLambda | LetRecLambda String String ProgLambda ProgLambda 
                | App App deriving Generic

data App = SimpleApp String Expr | ComplexApp App Expr deriving Generic

instance Show App where
  show (SimpleApp s e) | is_value e = s ++ " " ++ show e
                       | otherwise = s ++ " (" ++ show e ++ ")"
  show (ComplexApp a e) | is_value e = show a ++ " " ++ show e
                        | otherwise = show a ++ " (" ++ show e ++ ")"

instance Show ProgLambda where
  show UnitLambda = "unit"
  show (IfL c p1 p2) = "if " ++ show c ++ " then " ++ show p1 ++ " else " ++ show p2
  show (ExprL e) = show e
  show (LetLambda s p1 p2) = "let " ++ s ++ " = " ++ show p1 ++ " in " ++ show p2
  show (LetRecLambda s1 s2 p1 p2) = "letrec " ++ s1 ++ " " ++ s2 ++ " = " ++ show p1 ++ " in " ++ show p2
  show (App a) = show a

instance ToJSON App where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON App

instance ToJSON ProgLambda where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON ProgLambda

appP :: Parser App
appP = do var <- identifier <|> parens identifier
          (e1:e_rest) <- some expr
          return $ foldl ComplexApp (SimpleApp var e1) e_rest 
       <|> do firstapp <- parens appP
              apps <- some expr
              return $ foldl ComplexApp firstapp apps


progP, unitP, ifP, letP, letrecP :: Parser ProgLambda
progP = unitP <|> ifP <|> letP <|> letrecP <|> try (App <$> appP) <|> try (ExprL <$> expr) <|> parens progP

unitP = reserved "unit" *> pure UnitLambda

ifP = do
        reserved "if"
        cond <- cmp
        reserved "then"
        if_body <- progP
        reserved "else"
        else_body <- progP
        return $ IfL cond if_body else_body

letP = do
        reserved "let"
        var <- identifier
        equal
        value <- progP
        reserved "in"
        body <- progP
        return $ LetLambda var value body


letrecP = do
            reserved "letrec"
            var1 <- identifier
            var2 <- identifier
            equal
            value <- progP
            reserved "in"
            body <- progP
            return $ LetRecLambda var1 var2 value body

parseProg :: String -> Either ParseError ProgLambda
parseProg = parse (whiteSpace *> progP) ""
