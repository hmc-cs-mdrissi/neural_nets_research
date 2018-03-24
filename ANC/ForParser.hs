{-# LANGUAGE DeriveGeneric #-}

module ForParser where

import Text.Parsec.Prim ((<|>), parse)
import Text.Parsec.Combinator (chainl1)
import Text.Parsec.String (Parser)
import Text.Parsec.Error (ParseError)
import Data.Aeson

import Control.Applicative (liftA2)
import GHC.Generics

import ForLambdaCommon

data ProgFor = Assign String Expr | If Cmp ProgFor ProgFor | For String Expr Cmp Expr ProgFor
              | Seq ProgFor ProgFor deriving Generic

showProgFor :: ProgFor -> String
showProgFor (Assign s e) = s ++ " = " ++ show e ++ "\n"
showProgFor (If c p1 p2) = "if " ++ show c ++ "\nthen " ++ showProgFor p1 ++ " else " ++ showProgFor p2 ++ "\nendif\n"
showProgFor (For s e1 c e2 p1) = "for " ++ s ++ " = " ++ show e1 ++ "; " ++ show c ++ "; " ++ show e2 ++ " do \n" ++ showProgFor p1 ++ "\nendfor\n"
showProgFor (Seq p1 p2) = showProgFor p1 ++ "; " ++ showProgFor p2

instance Show ProgFor where
  show = showProgFor

instance ToJSON ProgFor where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON ProgFor

progP, progP_term, ifP, assignP, forP :: Parser ProgFor
progP = progP_term `chainl1` (semicolon *> pure Seq)

ifP = do
        reserved "if"
        cond <- cmp
        reserved "then"
        if_body <- progP
        reserved "else"
        else_body <- progP
        reserved "endif"
        return $ If cond if_body else_body

assignP = liftA2 Assign (identifier <* equal) expr

forP = do
        reserved "for"
        var <- identifier
        equal
        e1 <- expr
        semicolon
        cond <- cmp
        semicolon
        e2 <- expr
        reserved "do"
        body <- progP
        reserved "endfor"
        return $ For var e1 cond e2 body

progP_term = forP <|> ifP <|> assignP <|> parens progP

parseProg :: String -> Either ParseError ProgFor
parseProg = parse (whiteSpace *> progP) ""
